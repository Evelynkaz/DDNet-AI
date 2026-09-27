// Oracle B: runs the REAL, unmodified DDNet 20.1 server game code (CGameContext + CGameWorld +
// CCharacter + controller/teams -- everything CServer::Run() would tick for a live game: freeze,
// deep/live freeze, unfreeze, death, teleports, speedups, stoppers, switch/door layer, tune
// zones, endless hook, jump tiles, NPC/NPH/HIT tiles, solo/teams, hammer and other weapons,
// pickups, respawn) IN-PROCESS, without networking, driven tick by tick by scenario/generator
// inputs, and writes trace-b v1 files with the complete per-character state after every tick.
//
// This is DDNet-AI's own original file (GPL-3.0-only, like the rest of this repository) -- it
// contains no DDNet source. It is built against DDNet 20.1's *actual* engine/game object code
// (fetched, patched-in-a-copy and compiled by ../build-server-oracle.sh into a directory outside
// this repository -- see that script and tools/ddnet-oracle/README.md).
//
// Approach mirrors DDNet's own src/test/gameworld_test.cpp (creates CServer/kernel/storage/
// console/config/http/antibot/game server, loads a map, creates players, force-spawns them,
// calls OnTick) -- see docs/formats.md for the exact bootstrap/tick-loop sequence this
// reproduces, file:line references into the fetched tree, and the full trace-b v1 schema.
//
// Private-member access (task spec's sanctioned "well-contained technique", acceptance
// criterion 4): a handful of CCharacter/CCharacterCore fields the trace schema needs
// (m_ReloadTimer, m_AttackTick, m_QueuedWeapon, m_MoveRestrictions, ...) are `private` with no
// public getter. Rather than patch DDNet's headers (which would make this more than "a small
// CMake patch/overlay"), this translation unit redefines `private`/`protected` to `public`
// *before* including any DDNet header, and never after. Access specifiers do not affect class
// layout in any way (every mainstream C++ ABI, including the Itanium ABI these builds use,
// places non-virtual data members in declaration order regardless of the access section they
// fall in) -- object files compiled elsewhere (game-server-without-main etc.) see the real
// `private`/`protected` keywords as normal, so this is invisible to them and changes nothing
// about the binary layout they were compiled with; it only lifts the *compile-time* access
// check inside this one file.
//
// This requires every standard-library header to be fully included *before* the two `#define`s
// below take effect: libstdc++'s own <sstream> (transitively pulled in by <chrono>, which
// DDNet's own engine/http.h includes) relies on `private`/`protected` internally (a nested
// friend-declared struct in `std::basic_stringbuf`) and fails to parse a second time under the
// redefinition once already parsed once normally -- so every standard header this file (or
// anything it includes) needs is listed here, before the redefinition, priming their include
// guards so later transitive includes of the same headers are silently skipped.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define private public
#define protected public

#include <base/dbg.h>
#include <base/fs.h>
#include <base/hash.h>
#include <base/log.h>
#include <base/logger.h>
#include <base/str.h>

#include <engine/antibot.h>
#include <engine/config.h>
#include <engine/console.h>
#include <engine/engine.h>
#include <engine/http.h>
#include <engine/kernel.h>
#include <engine/map.h>
#include <engine/server.h>
#include <engine/server/databases/connection.h>
#include <engine/server/databases/connection_pool.h>
#include <engine/server/server.h>
#include <engine/shared/config.h>
#include <engine/shared/datafile.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <game/collision.h>
#include <game/gamecore.h>
#include <game/mapitems.h>
#include <game/server/entities/character.h>
#include <game/server/entities/door.h>
#include <game/server/entities/dragger.h>
#include <game/server/entities/dragger_beam.h>
#include <game/server/entities/gun.h>
#include <game/server/entities/laser.h>
#include <game/server/entities/light.h>
#include <game/server/entities/projectile.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>
#include <game/server/teams.h>
#include <game/version.h>

#include "../sha256.h"

// engine/server/server.h declares `bool IsInterrupted();` (checked once per outer Run() loop
// iteration for Ctrl-C) but never *defines* it -- the real server's src/engine/server/main.cpp
// and DDNet's own src/test/test.cpp each provide one definition (we link neither file: `main.cpp`
// is excluded from the `game-server-without-main` object library we link against by name, and
// `test.cpp` pulls in gtest, which this standalone harness has no other reason to need). We never
// install a signal handler and never loop on this ourselves (see RunTickLoop below, which does a
// fixed number of ticks and returns), so `false` is simply correct here, not a stub.
bool IsInterrupted()
{
	return false;
}

namespace
{

// =============================================================================================
// Small binary reader/writer -- field layout mirrors crates/ddai-trace/src/io.rs and
// tools/ddnet-oracle/oracle_core.cpp's own copy exactly (see docs/formats.md). Duplicated here
// (rather than shared via a header) on purpose: docs/formats.md documents the rawmap/scenario
// readers as independent per side/tool, and this harness is built completely separately from
// oracle_core.cpp (different translation unit, different link step).
// =============================================================================================

[[noreturn]] void Fail(const std::string &Msg)
{
	fprintf(stderr, "oracle_server: %s\n", Msg.c_str());
	exit(1);
}

class ByteReader
{
public:
	ByteReader(const unsigned char *Data, size_t Len) :
		m_pData(Data), m_Len(Len), m_Pos(0) {}

	const unsigned char *Take(size_t N, const char *Context)
	{
		if(m_Pos + N > m_Len)
			Fail(std::string("unexpected end of file while reading ") + Context);
		const unsigned char *P = m_pData + m_Pos;
		m_Pos += N;
		return P;
	}

	void ExpectMagic(const char Magic[4])
	{
		const unsigned char *P = Take(4, "magic");
		if(memcmp(P, Magic, 4) != 0)
			Fail(std::string("bad magic, expected ") + std::string(Magic, 4));
	}

	uint8_t U8(const char *Ctx) { return Take(1, Ctx)[0]; }
	uint16_t U16(const char *Ctx)
	{
		const unsigned char *P = Take(2, Ctx);
		return uint16_t(uint16_t(P[0]) | (uint16_t(P[1]) << 8));
	}
	int16_t I16(const char *Ctx) { return (int16_t)U16(Ctx); }
	uint32_t U32(const char *Ctx)
	{
		const unsigned char *P = Take(4, Ctx);
		return uint32_t(P[0]) | (uint32_t(P[1]) << 8) | (uint32_t(P[2]) << 16) | (uint32_t(P[3]) << 24);
	}
	int32_t I32(const char *Ctx) { return (int32_t)U32(Ctx); }
	uint64_t U64(const char *Ctx)
	{
		uint64_t Lo = U32(Ctx);
		uint64_t Hi = U32(Ctx);
		return Lo | (Hi << 32);
	}
	float F32(const char *Ctx)
	{
		uint32_t Bits = U32(Ctx);
		float V;
		memcpy(&V, &Bits, 4);
		return V;
	}
	std::string String16(const char *Ctx)
	{
		uint16_t Len = U16(Ctx);
		const unsigned char *P = Take(Len, Ctx);
		return std::string((const char *)P, Len);
	}
	std::string String32(const char *Ctx)
	{
		uint32_t Len = U32(Ctx);
		const unsigned char *P = Take(Len, Ctx);
		return std::string((const char *)P, Len);
	}
	void Bytes(unsigned char *Out, size_t N, const char *Ctx)
	{
		memcpy(Out, Take(N, Ctx), N);
	}

private:
	const unsigned char *m_pData;
	size_t m_Len;
	size_t m_Pos;
};

class ByteWriter
{
public:
	void U8(uint8_t V) { m_Buf.push_back(V); }
	void U16(uint16_t V)
	{
		m_Buf.push_back((unsigned char)(V & 0xff));
		m_Buf.push_back((unsigned char)((V >> 8) & 0xff));
	}
	void U32(uint32_t V)
	{
		for(int i = 0; i < 4; i++)
			m_Buf.push_back((unsigned char)((V >> (8 * i)) & 0xff));
	}
	void I32(int32_t V) { U32((uint32_t)V); }
	void U64(uint64_t V)
	{
		U32((uint32_t)(V & 0xffffffffULL));
		U32((uint32_t)(V >> 32));
	}
	void F32(float V)
	{
		uint32_t Bits;
		memcpy(&Bits, &V, 4);
		U32(Bits);
	}
	void Bytes(const void *P, size_t N)
	{
		const unsigned char *B = (const unsigned char *)P;
		m_Buf.insert(m_Buf.end(), B, B + N);
	}
	void Magic(const char M[4]) { Bytes(M, 4); }
	void String16(const std::string &S)
	{
		U16((uint16_t)S.size());
		Bytes(S.data(), S.size());
	}
	void String32(const std::string &S)
	{
		U32((uint32_t)S.size());
		Bytes(S.data(), S.size());
	}
	const std::vector<unsigned char> &Data() const { return m_Buf; }

private:
	std::vector<unsigned char> m_Buf;
};

std::vector<unsigned char> ReadFile(const std::string &Path)
{
	std::ifstream F(Path, std::ios::binary);
	if(!F)
		Fail("cannot open " + Path);
	std::vector<unsigned char> Data((std::istreambuf_iterator<char>(F)), std::istreambuf_iterator<char>());
	return Data;
}

void WriteFile(const std::string &Path, const std::vector<unsigned char> &Data)
{
	std::ofstream F(Path, std::ios::binary);
	if(!F)
		Fail("cannot open " + Path + " for writing");
	F.write((const char *)Data.data(), (std::streamsize)Data.size());
}

std::string JsonEscape(const std::string &S)
{
	std::string Out;
	Out.reserve(S.size());
	for(char C : S)
	{
		switch(C)
		{
		case '"': Out += "\\\""; break;
		case '\\': Out += "\\\\"; break;
		case '\n': Out += "\\n"; break;
		default: Out += C;
		}
	}
	return Out;
}

// =============================================================================================
// rawmap v1 reader (see docs/formats.md section 1) -- read-only, used for two things:
//   (a) sha256-checking a scenario's declared map_ref against the bytes we actually read;
//   (b) raw2map (below): converting the tile layers into a real, on-disk DDNet 20.1 `.map` file
//       via CDataFileWriter, so the harness can run recipe/rawmap-based scenarios (task 1.2's
//       synthetic recipes) through the real, unmodified server the same way as a real map.
// =============================================================================================

struct RawMap
{
	uint32_t Width = 0, Height = 0;
	std::vector<CTile> Game;
	bool HasFront = false;
	std::vector<CTile> Front;
	bool HasTele = false;
	std::vector<CTeleTile> Tele;
	bool HasSpeedup = false;
	std::vector<CSpeedupTile> Speedup;
	bool HasSwitch = false;
	std::vector<CSwitchTile> Switch;
	bool HasTune = false;
	std::vector<CTuneTile> Tune;
	// F5 (round-2 review): map "Settings" strings (console commands baked into the map's own
	// MAPITEMTYPE_INFO item, executed by `CGameContext::LoadMapSettings()` -- tune_zone/
	// switch_open/etc.) -- was previously always empty on the write side and never parsed on
	// the read side (round-1 comment here said "unused by this harness"), which meant every
	// map2raw'd rawmap silently dropped these commands, and every raw2map'd real-map replay
	// diverged from tick 0 on any map that has any (confirmed: BlmapChill's own map settings
	// include `tune_zone 1 ...`/`tune_zone 2 ...`/`switch_open 2/4/5/6/8/...`). See
	// ExtractRawMap/WriteMapFromRawMap below and docs/formats.md section 1/9.1.1.
	std::vector<std::string> Settings;
};

std::vector<CTile> ReadTiles(ByteReader &R, size_t N)
{
	std::vector<CTile> Out(N);
	for(size_t i = 0; i < N; i++)
	{
		Out[i].m_Index = R.U8("tile index");
		Out[i].m_Flags = R.U8("tile flags");
		Out[i].m_Skip = R.U8("tile skip");
		Out[i].m_MustBe0 = R.U8("tile reserved");
	}
	return Out;
}

RawMap ParseRawMap(const std::vector<unsigned char> &Bytes)
{
	ByteReader R(Bytes.data(), Bytes.size());
	R.ExpectMagic("RMP1");
	uint32_t Version = R.U32("version");
	if(Version != 1)
		Fail("unsupported rawmap version " + std::to_string(Version));
	RawMap M;
	M.Width = R.U32("width");
	M.Height = R.U32("height");
	uint8_t Present = R.U8("present bitmask");
	size_t N = (size_t)M.Width * (size_t)M.Height;

	M.Game = ReadTiles(R, N);
	if(Present & (1 << 0))
	{
		M.HasFront = true;
		M.Front = ReadTiles(R, N);
	}
	if(Present & (1 << 1))
	{
		M.HasTele = true;
		M.Tele.resize(N);
		for(size_t i = 0; i < N; i++)
		{
			M.Tele[i].m_Number = R.U8("tele number");
			M.Tele[i].m_Type = R.U8("tele type");
		}
	}
	if(Present & (1 << 2))
	{
		M.HasSpeedup = true;
		M.Speedup.resize(N);
		for(size_t i = 0; i < N; i++)
		{
			M.Speedup[i].m_Force = R.U8("speedup force");
			M.Speedup[i].m_MaxSpeed = R.U8("speedup max speed");
			M.Speedup[i].m_Type = R.U8("speedup type");
			R.U8("speedup reserved");
			M.Speedup[i].m_Angle = R.I16("speedup angle");
		}
	}
	if(Present & (1 << 3))
	{
		M.HasSwitch = true;
		M.Switch.resize(N);
		for(size_t i = 0; i < N; i++)
		{
			M.Switch[i].m_Number = R.U8("switch number");
			M.Switch[i].m_Type = R.U8("switch type");
			M.Switch[i].m_Flags = R.U8("switch flags");
			M.Switch[i].m_Delay = R.U8("switch delay");
		}
	}
	if(Present & (1 << 4))
	{
		M.HasTune = true;
		M.Tune.resize(N);
		for(size_t i = 0; i < N; i++)
		{
			M.Tune[i].m_Number = R.U8("tune number");
			M.Tune[i].m_Type = R.U8("tune type");
		}
	}
	uint32_t SettingsCount = R.U32("settings count");
	for(uint32_t i = 0; i < SettingsCount; i++)
		M.Settings.push_back(R.String32("setting"));
	return M;
}

void WriteRawMapFile(const RawMap &M, const std::string &Path)
{
	ByteWriter W;
	W.Magic("RMP1");
	W.U32(1);
	W.U32(M.Width);
	W.U32(M.Height);
	uint8_t Present = 0;
	if(M.HasFront)
		Present |= (1 << 0);
	if(M.HasTele)
		Present |= (1 << 1);
	if(M.HasSpeedup)
		Present |= (1 << 2);
	if(M.HasSwitch)
		Present |= (1 << 3);
	if(M.HasTune)
		Present |= (1 << 4);
	W.U8(Present);

	auto WriteTiles = [&](const std::vector<CTile> &V) {
		for(const CTile &T : V)
		{
			W.U8(T.m_Index);
			W.U8(T.m_Flags);
			W.U8(T.m_Skip);
			W.U8(T.m_MustBe0);
		}
	};
	WriteTiles(M.Game);
	if(M.HasFront)
		WriteTiles(M.Front);
	if(M.HasTele)
		for(const CTeleTile &T : M.Tele)
		{
			W.U8(T.m_Number);
			W.U8(T.m_Type);
		}
	if(M.HasSpeedup)
		for(const CSpeedupTile &T : M.Speedup)
		{
			W.U8(T.m_Force);
			W.U8(T.m_MaxSpeed);
			W.U8(T.m_Type);
			W.U8(0); // reserved (m_MustBe0)
			W.U16((uint16_t)T.m_Angle);
		}
	if(M.HasSwitch)
		for(const CSwitchTile &T : M.Switch)
		{
			W.U8(T.m_Number);
			W.U8(T.m_Type);
			W.U8(T.m_Flags);
			W.U8(T.m_Delay);
		}
	if(M.HasTune)
		for(const CTuneTile &T : M.Tune)
		{
			W.U8(T.m_Number);
			W.U8(T.m_Type);
		}
	W.U32((uint32_t)M.Settings.size()); // F5: was always 0 -- see RawMap::Settings's comment.
	for(const std::string &Setting : M.Settings)
		W.String32(Setting);
	WriteFile(Path, W.Data());
}

// map2raw (F5, round-1 review): the inverse of raw2map, extracting the tile layers CCollision
// already holds in memory (after LoadMap()) back into a rawmap v1 file. Lets a real-map-mode
// run emit a self-contained (rawmap + scenario v3) pair that replays byte-identically without
// the original `.map` file OR this harness's own real-map generator (see main()'s
// "--emit-scenario-v3" and docs/formats.md section 9.6) -- the whole point of task 1.6 being
// able to replay these traces without needing to run this C++ harness again.
// `pMap` is the loaded map to read the "Settings" item from (F5) -- a SEPARATE parameter from
// `pCol` because settings live in the map's own `MAPITEMTYPE_INFO` item, not in anything
// `CCollision`/`CLayers` expose (those only ever touch tile layers).
RawMap ExtractRawMap(CCollision *pCol, IMap *pMap)
{
	RawMap M;
	M.Width = (uint32_t)pCol->GetWidth();
	M.Height = (uint32_t)pCol->GetHeight();
	size_t N = (size_t)M.Width * (size_t)M.Height;

	const CTile *pGame = pCol->GameLayer();
	if(!pGame)
		Fail("map2raw: collision has no game layer (should be impossible for a loaded map)");
	M.Game.assign(pGame, pGame + N);

	if(const CTile *pFront = pCol->FrontLayer())
	{
		M.HasFront = true;
		M.Front.assign(pFront, pFront + N);
	}
	if(const CTeleTile *pTele = pCol->TeleLayer())
	{
		M.HasTele = true;
		M.Tele.assign(pTele, pTele + N);
	}
	if(const CSpeedupTile *pSpeedup = pCol->SpeedupLayer())
	{
		M.HasSpeedup = true;
		M.Speedup.assign(pSpeedup, pSpeedup + N);
	}
	if(const CSwitchTile *pSwitch = pCol->SwitchLayer())
	{
		M.HasSwitch = true;
		M.Switch.assign(pSwitch, pSwitch + N);
	}
	if(const CTuneTile *pTune = pCol->TuneLayer())
	{
		M.HasTune = true;
		M.Tune.assign(pTune, pTune + N);
	}

	// F5: extract the map's own "Settings" strings -- mirrors `CGameContext::LoadMapSettings()`
	// (gamecontext.cpp) reading `MAPITEMTYPE_INFO`, item id 0, `CMapItemInfoSettings::m_Settings`
	// (a data index for a blob of NUL-terminated strings back to back). Read-only here (we
	// never call `UnloadData` -- this harness never reloads the map, so there is nothing to
	// free for again later, and `IMap::UnloadData` on this "temporary .map" `CMap` would be a
	// no-op past process exit anyway).
	{
		int Start = 0, Num = 0;
		pMap->GetType(MAPITEMTYPE_INFO, &Start, &Num);
		for(int i = Start; i < Start + Num; i++)
		{
			int ItemId = 0;
			auto *pItem = (CMapItemInfoSettings *)pMap->GetItem(i, nullptr, &ItemId, nullptr);
			int ItemSize = pMap->GetItemSize(i);
			if(!pItem || ItemId != 0 || ItemSize < (int)sizeof(CMapItemInfoSettings) || pItem->m_Settings <= -1)
				continue;
			int Size = pMap->GetDataSize(pItem->m_Settings);
			const char *pSettings = (const char *)pMap->GetData(pItem->m_Settings);
			const char *pNext = pSettings;
			while(pNext < pSettings + Size)
			{
				M.Settings.emplace_back(pNext);
				pNext += M.Settings.back().size() + 1;
			}
			break;
		}
	}
	return M;
}

// raw2map: writes `RawMap` as a real DDNet 20.1 datafile-v4 `.map` (task spec acceptance
// criterion 2 -- "the harness converts to a temporary .map using DDNet 20.1's own
// CDataFileWriter + mapitems"). Layer/group population mirrors tools/ddnet-oracle/
// oracle_core.cpp's `BuildCollision`/`MakeTilemapItem` field-for-field (the same values, just
// written through `CDataFileWriter::AddItem`/`AddData` to real datafile bytes instead of an
// in-memory `IMap` override) -- see src/tools/dummy_map.cpp in the fetched DDNet tree for the
// reference pattern this follows (Open/AddItem/AddData/Finish). `CLayers::InitTilemapSkip`
// (src/game/layers.cpp) documents that non-game tile layers (tele/speedup/front/switch/tune)
// don't need (and don't validate) a `m_Data` tile array of their own -- only their dedicated
// `m_Tele`/`m_Speedup`/`m_Switch`/`m_Tune` index is read -- so, like oracle_core.cpp, this
// leaves `m_Data == -1` for them.
CMapItemLayerTilemap MakeTilemapItem(int Width, int Height, int Flags, const char *pName)
{
	CMapItemLayerTilemap L{};
	L.m_Layer.m_Type = LAYERTYPE_TILES;
	L.m_Layer.m_Flags = 0;
	L.m_Version = 3;
	L.m_Width = Width;
	L.m_Height = Height;
	L.m_Flags = Flags;
	L.m_Color = {255, 255, 255, 255};
	L.m_ColorEnv = -1;
	L.m_ColorEnvOffset = 0;
	L.m_Image = -1;
	L.m_Data = -1;
	L.m_Tele = -1;
	L.m_Speedup = -1;
	L.m_Front = -1;
	L.m_Switch = -1;
	L.m_Tune = -1;
	// `EnsureTileLayerProperties` (engine/shared/map.cpp) requires `m_aName` to decode (via
	// `IntsToStr`) to valid UTF-8 -- an all-zero name decodes each byte to -128 (signed char),
	// which is not valid UTF-8 continuation-byte-first and fails to load at all. A real name is
	// needed (map.cpp silently *corrects* a mismatched-but-valid name for the physics layers, it
	// only *rejects* the file outright for a name that doesn't decode as UTF-8 at all).
	StrToInts(L.m_aName, std::size(L.m_aName), pName);
	return L;
}

bool WriteMapFromRawMap(const RawMap &Map, IStorage *pStorage, const std::string &MapName)
{
	std::string Path = "maps/" + MapName + ".map";
	CDataFileWriter Writer;
	if(!Writer.Open(pStorage, Path.c_str()))
		return false;

	// `CMap::ValidateMapVersion` (engine/shared/map.cpp) requires a MAPITEMTYPE_VERSION item
	// with `m_Version == 1` before it will load anything else in the file -- dummy_map.cpp (the
	// reference this function otherwise follows) never round-trips its own output back through
	// the real map loader, so it happens not to need this, but a map LoadMap() must open does.
	CMapItemVersion VersionItem{};
	VersionItem.m_Version = 1;
	Writer.AddItem(MAPITEMTYPE_VERSION, 0, sizeof(VersionItem), &VersionItem);

	// F5 (round-2 review): write the map's own "Settings" strings into a MAPITEMTYPE_INFO item
	// so `CGameContext::LoadMapSettings()` (gamecontext.cpp) executes them at the normal point
	// during `OnInit()` -- exactly the commands a real load of the original map would run
	// (tune_zone/switch_open/etc.). Previously omitted entirely (round-1's raw2map never wrote
	// an INFO item at all), which silently dropped these on every real-map round trip.
	if(!Map.Settings.empty())
	{
		std::vector<char> Blob;
		for(const std::string &Setting : Map.Settings)
		{
			Blob.insert(Blob.end(), Setting.begin(), Setting.end());
			Blob.push_back('\0');
		}
		int SettingsData = Writer.AddData(Blob.size(), Blob.data());
		CMapItemInfoSettings InfoItem{};
		InfoItem.m_Version = 1;
		InfoItem.m_Author = -1;
		InfoItem.m_MapVersion = -1;
		InfoItem.m_Credits = -1;
		InfoItem.m_License = -1;
		InfoItem.m_Settings = SettingsData;
		Writer.AddItem(MAPITEMTYPE_INFO, 0, sizeof(InfoItem), &InfoItem);
	}

	int NumLayers = 1;
	if(Map.HasFront)
		NumLayers++;
	if(Map.HasTele)
		NumLayers++;
	if(Map.HasSpeedup)
		NumLayers++;
	if(Map.HasSwitch)
		NumLayers++;
	if(Map.HasTune)
		NumLayers++;

	CMapItemGroup Group{};
	Group.m_Version = 3;
	Group.m_ParallaxX = Group.m_ParallaxY = 100;
	Group.m_StartLayer = 0;
	Group.m_NumLayers = NumLayers;
	Writer.AddItem(MAPITEMTYPE_GROUP, 0, sizeof(Group), &Group);

	CMapItemLayerTilemap GameL = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_GAME, "Game");
	GameL.m_Data = Writer.AddData(Map.Game.size() * sizeof(CTile), Map.Game.data());
	Writer.AddItem(MAPITEMTYPE_LAYER, 0, sizeof(GameL), &GameL);

	if(Map.HasFront)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_FRONT, "Front");
		L.m_Front = Writer.AddData(Map.Front.size() * sizeof(CTile), Map.Front.data());
		Writer.AddItem(MAPITEMTYPE_LAYER, 1, sizeof(L), &L);
	}
	if(Map.HasTele)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_TELE, "Tele");
		L.m_Tele = Writer.AddData(Map.Tele.size() * sizeof(CTeleTile), Map.Tele.data());
		Writer.AddItem(MAPITEMTYPE_LAYER, 2, sizeof(L), &L);
	}
	if(Map.HasSpeedup)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_SPEEDUP, "Speedup");
		L.m_Speedup = Writer.AddData(Map.Speedup.size() * sizeof(CSpeedupTile), Map.Speedup.data());
		Writer.AddItem(MAPITEMTYPE_LAYER, 3, sizeof(L), &L);
	}
	if(Map.HasSwitch)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_SWITCH, "Switch");
		L.m_Switch = Writer.AddData(Map.Switch.size() * sizeof(CSwitchTile), Map.Switch.data());
		Writer.AddItem(MAPITEMTYPE_LAYER, 4, sizeof(L), &L);
	}
	if(Map.HasTune)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_TUNE, "Tune");
		L.m_Tune = Writer.AddData(Map.Tune.size() * sizeof(CTuneTile), Map.Tune.data());
		Writer.AddItem(MAPITEMTYPE_LAYER, 5, sizeof(L), &L);
	}

	Writer.Finish();
	return true;
}

// =============================================================================================
// scenario v2 reader (docs/formats.md section 2) -- read-only, byte-for-byte the same format
// Oracle A reads (tools/ddnet-oracle/oracle_core.cpp): unchanged, so a scenario produced for
// Oracle A can be fed to Oracle B verbatim for the parity check (docs/formats.md /
// acceptance criterion 7). `ResolveInput` is the same "aim mode" algorithm, independently
// re-implemented (per docs/formats.md section 2.1's own note that both sides implement it
// separately from the same spec, not by calling into each other).
// =============================================================================================

struct TuningOverride
{
	std::string Name;
	int32_t ValueX100 = 0;
};

struct CharacterSpawnRaw
{
	uint32_t Id = 0;
	int32_t SpawnX = 0, SpawnY = 0;
	int32_t Team = 0; // v3 only (F5); always 0 for a v2 file.
};

struct ScenarioInputRecord
{
	int32_t Direction = 0, TargetX = 0, TargetY = 0, AimSlot = -1, Jump = 0, Fire = 0, Hook = 0;
	int32_t PlayerFlags = 0, WantedWeapon = 0, NextWeapon = 0, PrevWeapon = 0;
	// F12 (round-2 review): v3-only 12th field -- a recorded "kill" request, applied exactly
	// like a real client's `CNetMsg_Cl_Kill` (see main()'s tick loop and docs/formats.md
	// section 9.6). Always 0 for a v2 file (that reader never touches this field at all).
	int32_t Kill = 0;
};

// F5 (round-1 review): scenario v3 is an ADDITIVE superset of v2 (same magic "SCN1", version
// bumped 2 -> 3) -- adds a per-character `team` and a trailer (cfg command lines, generator id,
// seed) so a real-map-mode run's exact resolved inputs are replayable from disk alone, without
// the C++ harness's own generator or the original `.map` file (see WriteScenarioV3/map2raw
// below and docs/formats.md section 9.6). v2 files (`version == 2`, Oracle A's own format,
// task 1.2 -- untouched) parse with `Team == 0` for every character and empty trailer fields.
struct Scenario
{
	uint32_t Version = 2;
	bool MapRefIsRecipe = true;
	std::string MapRefString;
	std::array<unsigned char, 32> MapSha256{};
	bool NoWeakHook = false;
	std::vector<TuningOverride> TuningOverrides;
	std::vector<CharacterSpawnRaw> Characters;
	std::vector<std::vector<ScenarioInputRecord>> Inputs; // Inputs[tick][slot]
	// v3-only trailer:
	std::vector<std::string> CfgLines; // executed at both the pre-init and post-init points,
	// exactly like a `--cfg` file (see the F1 fix's comment in main()) -- not split into
	// separate pre/post lists since this harness always runs the same content at both points.
	std::string GeneratorId;
	uint64_t EmbeddedSeed = 0;
};

void ValidateScenario(const Scenario &S)
{
	for(const auto &C : S.Characters)
	{
		if(C.Id >= (uint32_t)MAX_CLIENTS)
			Fail("character id " + std::to_string(C.Id) + " must be < MAX_CLIENTS (" + std::to_string(MAX_CLIENTS) + ")");
	}
	for(size_t i = 0; i < S.Characters.size(); i++)
		for(size_t j = 0; j < i; j++)
			if(S.Characters[i].Id == S.Characters[j].Id)
				Fail("duplicate character id " + std::to_string(S.Characters[i].Id));
	int32_t NumChars = (int32_t)S.Characters.size();
	for(const auto &Tick : S.Inputs)
		for(const auto &In : Tick)
			if(In.AimSlot < -1 || In.AimSlot >= NumChars)
				Fail("aim_slot " + std::to_string(In.AimSlot) + " must be -1 or a valid character slot index");
}

Scenario ParseScenario(const std::vector<unsigned char> &Bytes)
{
	ByteReader R(Bytes.data(), Bytes.size());
	R.ExpectMagic("SCN1");
	uint32_t Version = R.U32("version");
	if(Version != 2 && Version != 3)
		Fail("unsupported scenario version " + std::to_string(Version) + " (expected 2 or 3)");
	Scenario S;
	S.Version = Version;
	uint8_t Tag = R.U8("map ref tag");
	S.MapRefIsRecipe = (Tag == 0);
	S.MapRefString = R.String16("map ref string");
	R.Bytes(S.MapSha256.data(), 32, "map sha256");
	S.NoWeakHook = R.U8("no_weak_hook") != 0;

	uint32_t OverrideCount = R.U32("tuning override count");
	for(uint32_t i = 0; i < OverrideCount; i++)
	{
		TuningOverride O;
		O.Name = R.String16("tuning override name");
		O.ValueX100 = R.I32("tuning override value");
		S.TuningOverrides.push_back(O);
	}

	uint32_t CharCount = R.U32("character count");
	for(uint32_t i = 0; i < CharCount; i++)
	{
		CharacterSpawnRaw C;
		C.Id = R.U32("character id");
		C.SpawnX = R.I32("character spawn x");
		C.SpawnY = R.I32("character spawn y");
		if(Version >= 3)
			C.Team = R.I32("character team");
		S.Characters.push_back(C);
	}

	uint32_t TickCount = R.U32("tick count");
	S.Inputs.resize(TickCount);
	for(uint32_t t = 0; t < TickCount; t++)
	{
		S.Inputs[t].resize(CharCount);
		for(uint32_t c = 0; c < CharCount; c++)
		{
			ScenarioInputRecord &In = S.Inputs[t][c];
			In.Direction = R.I32("input direction");
			In.TargetX = R.I32("input target_x");
			In.TargetY = R.I32("input target_y");
			In.AimSlot = R.I32("input aim_slot");
			In.Jump = R.I32("input jump");
			In.Fire = R.I32("input fire");
			In.Hook = R.I32("input hook");
			In.PlayerFlags = R.I32("input player_flags");
			In.WantedWeapon = R.I32("input wanted_weapon");
			In.NextWeapon = R.I32("input next_weapon");
			In.PrevWeapon = R.I32("input prev_weapon");
			if(Version >= 3)
				In.Kill = R.I32("input kill");
		}
	}

	if(Version >= 3)
	{
		uint32_t CfgLineCount = R.U32("cfg line count");
		for(uint32_t i = 0; i < CfgLineCount; i++)
			S.CfgLines.push_back(R.String16("cfg line"));
		S.GeneratorId = R.String16("generator id");
		S.EmbeddedSeed = R.U64("embedded seed");
	}

	ValidateScenario(S);
	return S;
}

struct ResolvedInput
{
	int32_t Direction = 0, TargetX = 0, TargetY = 0, Jump = 0, Fire = 0, Hook = 0;
	int32_t PlayerFlags = 0, WantedWeapon = 0, NextWeapon = 0, PrevWeapon = 0;
	int32_t Kill = 0; // F12: see ScenarioInputRecord::Kill.
};

ResolvedInput ResolveInput(const ScenarioInputRecord &In, size_t SelfSlot, const std::vector<std::pair<int32_t, int32_t>> &PrevPositions)
{
	int32_t TargetX, TargetY;
	if(In.AimSlot >= 0)
	{
		size_t K = (size_t)In.AimSlot;
		int32_t Bx = PrevPositions[K].first - PrevPositions[SelfSlot].first + In.TargetX;
		int32_t By = PrevPositions[K].second - PrevPositions[SelfSlot].second + In.TargetY;
		if(Bx == 0 && By == 0)
		{
			TargetX = In.TargetX;
			TargetY = In.TargetY;
		}
		else
		{
			TargetX = Bx;
			TargetY = By;
		}
	}
	else
	{
		TargetX = In.TargetX;
		TargetY = In.TargetY;
	}
	ResolvedInput Out;
	Out.Direction = In.Direction;
	Out.TargetX = TargetX;
	Out.TargetY = TargetY;
	Out.Jump = In.Jump;
	Out.Fire = In.Fire;
	Out.Hook = In.Hook;
	Out.PlayerFlags = In.PlayerFlags;
	Out.WantedWeapon = In.WantedWeapon;
	Out.NextWeapon = In.NextWeapon;
	Out.PrevWeapon = In.PrevWeapon;
	Out.Kill = In.Kill; // 0 for every v2 file -- ScenarioInputRecord::Kill is never populated by v2 parsing.
	return Out;
}

// Writer side of scenario v3 (F5) -- used by real-map mode to emit a self-contained,
// byte-exact replay of whatever it just resolved and ran (see main()'s "--emit-scenario-v3").
void WriteScenarioV3(const std::string &Path, const std::string &RawmapPath, const std::array<unsigned char, 32> &RawmapSha256,
	const std::vector<CharacterSpawnRaw> &Characters, const std::vector<std::vector<ResolvedInput>> &AllInputs,
	const std::vector<std::string> &CfgLines, const std::string &GeneratorId, uint64_t Seed)
{
	ByteWriter W;
	W.Magic("SCN1");
	W.U32(3);
	W.U8(1); // map_ref_tag = RawmapFile
	W.String16(RawmapPath);
	W.Bytes(RawmapSha256.data(), 32);
	W.U8(0); // no_weak_hook -- superseded by CfgLines for v3
	W.U32(0); // tuning_override_count -- superseded by CfgLines for v3
	W.U32((uint32_t)Characters.size());
	for(const auto &C : Characters)
	{
		W.U32(C.Id);
		W.I32(C.SpawnX);
		W.I32(C.SpawnY);
		W.I32(C.Team);
	}
	W.U32((uint32_t)AllInputs.size());
	for(const auto &TickRow : AllInputs)
	{
		for(const auto &In : TickRow)
		{
			W.I32(In.Direction);
			W.I32(In.TargetX);
			W.I32(In.TargetY);
			W.I32(-1); // aim_slot: already resolved, always explicit
			W.I32(In.Jump);
			W.I32(In.Fire);
			W.I32(In.Hook);
			W.I32(In.PlayerFlags);
			W.I32(In.WantedWeapon);
			W.I32(In.NextWeapon);
			W.I32(In.PrevWeapon);
			W.I32(In.Kill);
		}
	}
	W.U32((uint32_t)CfgLines.size());
	for(const auto &Line : CfgLines)
		W.String16(Line);
	W.String16(GeneratorId);
	W.U64(Seed);
	WriteFile(Path, W.Data());
}

// =============================================================================================
// SplitMix64 (public domain, Sebastiano Vigna) -- byte-for-byte the same algorithm documented in
// docs/formats.md section 4 and implemented independently in Rust (crates/ddai-trace/src/prng.rs)
// for `random-v1`. This is a SEPARATE C++ re-implementation for the harness-side real-map
// generator (docs/formats.md section 9, acceptance criterion 6's "harness-side generator"
// alternative) -- not bit-for-bit tested against the Rust one (different generator, different
// scenarios by construction -- real maps vs synthetic recipes), but the same well-specified
// algorithm.
// =============================================================================================

class SplitMix64
{
public:
	explicit SplitMix64(uint64_t Seed) :
		m_State(Seed) {}

	uint64_t NextU64()
	{
		m_State += 0x9E3779B97F4A7C15ULL;
		uint64_t Z = m_State;
		Z = (Z ^ (Z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		Z = (Z ^ (Z >> 27)) * 0x94D049BB133111EBULL;
		return Z ^ (Z >> 31);
	}

	// Multiply-high (Lemire), same as crates/ddai-trace/src/prng.rs's `below`.
	uint32_t Below(uint32_t Bound)
	{
		if(Bound == 0)
			return 0;
		return (uint32_t)(((unsigned __int128)NextU64() * (unsigned __int128)Bound) >> 64);
	}

	int32_t RangeInclusive(int32_t Lo, int32_t Hi)
	{
		return Lo + (int32_t)Below((uint32_t)(Hi - Lo + 1));
	}

	// Uniform in [-1, 0, 1], matching random-v1's direction redraw.
	int32_t Sign()
	{
		return (int32_t)Below(3) - 1;
	}

	bool Chance(uint32_t Numerator, uint32_t Denominator)
	{
		return Below(Denominator) < Numerator;
	}

private:
	uint64_t m_State;
};

} // namespace

// =============================================================================================
// Harness-side real-map scenario generator (docs/formats.md section 9). Runs against the REAL,
// already-loaded `CCollision` (task spec acceptance criterion 6's "harness-side generator"
// alternative to extending `ddnet-ai trace gen-scenario`): spawns characters on free cells,
// preferring cells near freeze edges and close to each other, then drives a small deterministic
// per-character input state machine every tick (direction/jump/hook hold-release timers, aim
// permanently tracking the nearest other character with periodic noise, occasional hammer
// swings) -- see docs/formats.md for the exact algorithm this implements.
// =============================================================================================
namespace realmap_gen
{

bool IsFreezeRawIndex(int Idx)
{
	return Idx == TILE_FREEZE || Idx == TILE_DFREEZE || Idx == TILE_LFREEZE;
}

struct Cell
{
	int X, Y;
};

// Free cell: TILE_AIR on both the game layer and (if present) the front layer -- same
// definition `docs/formats.md` section 3 already uses for the synthetic recipes.
bool IsFreeCell(CCollision *pCollision, int X, int Y)
{
	return pCollision->GetIndex(X, Y) == TILE_AIR && pCollision->GetFrontIndex(X, Y) == TILE_AIR;
}

bool IsNearFreeze(CCollision *pCollision, int X, int Y, int Radius)
{
	int W = pCollision->GetWidth(), H = pCollision->GetHeight();
	for(int dy = -Radius; dy <= Radius; dy++)
	{
		for(int dx = -Radius; dx <= Radius; dx++)
		{
			int nx = X + dx, ny = Y + dy;
			if(nx < 0 || ny < 0 || nx >= W || ny >= H)
				continue;
			if(IsFreezeRawIndex(pCollision->GetIndex(nx, ny)) || IsFreezeRawIndex(pCollision->GetFrontIndex(nx, ny)))
				return true;
		}
	}
	return false;
}

// Coverage addition (round-2 review, orchestrator addition to F7): true if any cell within
// `Radius` (Chebyshev) of (X,Y) matches `Pred` -- used below with three SEPARATE predicates
// (teleport/switch/tune) rather than one combined one, specifically so a map whose switch
// layer covers far more tiles than its tune layer (BlmapChill: ~30 switch numbers over a large
// area vs. 8 small tune zones -- confirmed by inspecting its extracted rawmap) doesn't drown
// out the rarer category in one shared pool -- an earlier version of this function combined all
// three into a single `IsNearSpecial`, which produced a `SpecialPool` overwhelmingly biased
// toward whichever category has the most tiles; the first round-3 corpus built with that
// version got `tune_zone_ticks: 0` for BlmapChill across all 70 of its scenarios despite the
// bias mechanism existing at all, purely because the combined pool was picked from uniformly
// by CELL, not by category. `Index` for these `CCollision` accessors is the raw tile index
// (`Y*Width+X`) -- confirmed against this exact build's collision.cpp (`GetIndex(Nx,Ny)`
// returns `m_pTiles[Ny*Width+Nx]`, and `IsTeleport`/`IsTune`/`GetSwitchType` index their own
// parallel arrays the same way), so no pixel-coordinate round trip through `GetPureMapIndex` is
// needed since we already have tile (X,Y).
template<typename Pred>
bool IsNearTileMatching(CCollision *pCollision, int X, int Y, int Radius, Pred &&Matches)
{
	int W = pCollision->GetWidth(), H = pCollision->GetHeight();
	for(int dy = -Radius; dy <= Radius; dy++)
	{
		for(int dx = -Radius; dx <= Radius; dx++)
		{
			int nx = X + dx, ny = Y + dy;
			if(nx < 0 || ny < 0 || nx >= W || ny >= H)
				continue;
			if(Matches(ny * W + nx))
				return true;
		}
	}
	return false;
}

// Finds up to `Count` distinct free-cell spawn positions (pixel coordinates, tile centers).
// Mirrors docs/formats.md section 4's synthetic-recipe spawn algorithm: the first character
// prefers a free cell near a freeze edge (falls back to any free cell if the map has none within
// the search radius, e.g. an arena-like real map); each following character is, with probability
// 1/2, placed at Chebyshev distance 1..=3 from an already-placed character (so hook/hammer
// interactions are likely), otherwise independently random among the (near-freeze-preferring)
// pool; positions never repeat.
//
// Coverage addition (round-2 review): on top of the above, roughly 1 in 4 spawns (the "First"
// character, and each subsequent character that lands in the "independently random" branch
// rather than anchored next to an already-placed teammate) is instead drawn from one of three
// SEPARATE pools -- free cells within 3 tiles of a teleporter-in tile, of a switch tile, or of
// a tune-zone tile -- picked with EQUAL PROBABILITY PER NON-EMPTY CATEGORY (not per cell, see
// `IsNearTileMatching`'s comment above for why), when the map has any such cells at all (e.g.
// Blockdale has none of the three per the review's own research, so this is a no-op there and
// falls through to the pre-existing pools unchanged).
std::vector<Cell> ChooseSpawns(CCollision *pCollision, SplitMix64 &Rng, uint32_t Count)
{
	int W = pCollision->GetWidth(), H = pCollision->GetHeight();
	std::vector<Cell> Free, NearFreeze, NearTele, NearSwitch, NearTune;
	for(int y = 0; y < H; y++)
	{
		for(int x = 0; x < W; x++)
		{
			if(!IsFreeCell(pCollision, x, y))
				continue;
			Free.push_back({x, y});
			if(IsNearFreeze(pCollision, x, y, 3))
				NearFreeze.push_back({x, y});
			if(IsNearTileMatching(pCollision, x, y, 3, [&](int Idx) { return pCollision->IsTeleport(Idx) || pCollision->IsEvilTeleport(Idx); }))
				NearTele.push_back({x, y});
			if(IsNearTileMatching(pCollision, x, y, 3, [&](int Idx) { return pCollision->GetSwitchType(Idx) != 0; }))
				NearSwitch.push_back({x, y});
			if(IsNearTileMatching(pCollision, x, y, 3, [&](int Idx) { return pCollision->IsTune(Idx) != 0; }))
				NearTune.push_back({x, y});
		}
	}
	if(Free.empty())
		Fail("real map has no free (TILE_AIR) cells to spawn characters on");
	const std::vector<Cell> &PreferredPool = NearFreeze.empty() ? Free : NearFreeze;

	std::vector<const std::vector<Cell> *> SpecialCategories;
	for(const auto *Pool : {&NearTele, &NearSwitch, &NearTune})
		if(!Pool->empty())
			SpecialCategories.push_back(Pool);
	// One (uniformly, equal-weight) shared pick per spawn call below -- a lambda so both the
	// "First" character and the per-following-character branch use the exact same logic.
	auto PickSpecialOrNull = [&]() -> const std::vector<Cell> * {
		if(SpecialCategories.empty() || !Rng.Chance(1, 4))
			return nullptr;
		return SpecialCategories[Rng.Below((uint32_t)SpecialCategories.size())];
	};

	std::vector<Cell> Chosen;
	auto AlreadyChosen = [&](const Cell &C) {
		return std::find_if(Chosen.begin(), Chosen.end(), [&](const Cell &O) { return O.X == C.X && O.Y == C.Y; }) != Chosen.end();
	};

	const std::vector<Cell> *FirstSpecial = PickSpecialOrNull();
	Cell First = FirstSpecial ? (*FirstSpecial)[Rng.Below((uint32_t)FirstSpecial->size())]
				   : PreferredPool[Rng.Below((uint32_t)PreferredPool.size())];
	Chosen.push_back(First);

	for(uint32_t i = 1; i < Count; i++)
	{
		Cell Next{};
		bool Found = false;
		if(Rng.Chance(1, 2))
		{
			const Cell &Anchor = Chosen[Rng.Below((uint32_t)Chosen.size())];
			std::vector<Cell> Nearby;
			for(int dy = -3; dy <= 3; dy++)
				for(int dx = -3; dx <= 3; dx++)
				{
					if(dx == 0 && dy == 0)
						continue;
					if(std::max(std::abs(dx), std::abs(dy)) > 3)
						continue;
					int nx = Anchor.X + dx, ny = Anchor.Y + dy;
					if(nx < 0 || ny < 0 || nx >= W || ny >= H)
						continue;
					if(!IsFreeCell(pCollision, nx, ny))
						continue;
					Cell C{nx, ny};
					if(!AlreadyChosen(C))
						Nearby.push_back(C);
				}
			if(!Nearby.empty())
			{
				Next = Nearby[Rng.Below((uint32_t)Nearby.size())];
				Found = true;
			}
		}
		// Coverage addition (round-2 review): same ~1-in-4 special-tile bias as the "First"
		// character above, applied here too so it isn't only ever the very first spawn. Rolls
		// its OWN category pick (independent of the "First" character's) so different
		// characters in the same scenario can land near different feature categories.
		if(!Found)
		{
			if(const std::vector<Cell> *Special = PickSpecialOrNull())
			{
				for(int Attempt = 0; Attempt < 4096 && !Found; Attempt++)
				{
					Cell C = (*Special)[Rng.Below((uint32_t)Special->size())];
					if(!AlreadyChosen(C))
					{
						Next = C;
						Found = true;
					}
				}
			}
		}
		if(!Found)
		{
			for(int Attempt = 0; Attempt < 4096 && !Found; Attempt++)
			{
				Cell C = Free[Rng.Below((uint32_t)Free.size())];
				if(!AlreadyChosen(C))
				{
					Next = C;
					Found = true;
				}
			}
		}
		if(!Found)
			Fail("real map does not have enough distinct free cells for the requested character count");
		Chosen.push_back(Next);
	}
	return Chosen;
}

// Per-character input FSM state -- constants mirror docs/formats.md section 4's `random-v1`
// (direction/jump/hook hold-release timers) so both generators read as "the same style" of
// synthetic input, even though this is a separate implementation with its own aim/fire policy
// (permanent nearest-character targeting + occasional hammer swings, see docs/formats.md
// section 9) tailored to exercising block-style interactions on real maps.
struct CharGenState
{
	int32_t Direction = 0;
	uint32_t DirectionTicksLeft = 0;
	int32_t Jump = 0;
	uint32_t JumpTicksLeft = 0;
	bool JumpHeld = false;
	int32_t Hook = 0;
	uint32_t HookTicksLeft = 0;
	bool HookHeld = false;
	int32_t AimNoiseX = 0, AimNoiseY = 0;
	uint32_t AimNoiseTicksLeft = 0;
	// F7 (round-1 review): `m_Fire` is a monotonic edge counter masked to `INPUT_STATE_MASK`
	// (0x3f, generated/protocol.h) -- `HandleWeaponSwitch`/`FireWeapon` (character.cpp) read it
	// through `CountInput(Prev, Cur)`, which walks the counter forward from `Prev` to `Cur`
	// modulo 64 and counts how many times bit 0 (the "currently held" bit) flipped to 1 along
	// the way. The old code set `Fire` directly to a 0/1 level each tick; going from 1 back to
	// 0 the NEXT tick made `CountInput` walk all the way around (63 -> 0), reporting 31 presses
	// for a single release. `FireHeld`/`FireCounter` here model a real client instead: the
	// counter only changes ON a press/release edge (by exactly 1, matching a real client one
	// input packet per edge), never while simply holding or releasing.
	bool FireHeld = false;
	int32_t FireCounter = 0;
	int32_t WantedWeapon = 0; // 1-based (0 = no explicit switch request) -- see StepCharGen.
	// F12 (round-2 review): the generator's OWN policy for WHEN to request a kill (the harness
	// only applies the resulting bit, via the real `OnKillNetMessage` path -- see main()). Ticks
	// this character has been continuously frozen, as of the END of the PREVIOUS tick (the most
	// recent state available when this tick's input is being decided, exactly like a real
	// player deciding based on what they last observed).
	uint32_t ContinuousFrozenTicks = 0;
};

ResolvedInput StepCharGen(CharGenState &St, SplitMix64 &Rng, size_t Self, const std::vector<std::pair<int32_t, int32_t>> &Positions, bool WasFrozenLastTick)
{
	// Nearest other character (live position, recomputed every tick -- moved ahead of the
	// direction FSM below in the round-1 review fix, F7, so movement itself can close in on it,
	// not just aim/fire).
	int64_t NearestDx = 0, NearestDy = 0;
	int64_t BestDistSq = -1;
	for(size_t Other = 0; Other < Positions.size(); Other++)
	{
		if(Other == Self)
			continue;
		int64_t Dx = (int64_t)Positions[Other].first - Positions[Self].first;
		int64_t Dy = (int64_t)Positions[Other].second - Positions[Self].second;
		int64_t D2 = Dx * Dx + Dy * Dy;
		if(BestDistSq < 0 || D2 < BestDistSq)
		{
			BestDistSq = D2;
			NearestDx = Dx;
			NearestDy = Dy;
		}
	}

	// F7 (round-1 review): closed-loop block policy, part 1 -- bias movement direction toward
	// the nearest other character (2/3 of the time on a redraw) instead of a fully independent
	// random walk, so characters actually close the distance often enough to exercise
	// hammer/hook-on-player interactions, rather than relying on incidental proximity from pure
	// random wandering (the round-1 corpus averaged well under one hammer swing per scenario).
	if(St.DirectionTicksLeft == 0)
	{
		if(BestDistSq >= 0 && NearestDx != 0 && Rng.Chance(2, 3))
			St.Direction = NearestDx > 0 ? 1 : -1;
		else
			St.Direction = Rng.Sign();
		St.DirectionTicksLeft = (uint32_t)Rng.RangeInclusive(5, 40);
	}
	St.DirectionTicksLeft--;

	if(St.JumpTicksLeft == 0)
	{
		St.JumpHeld = !St.JumpHeld;
		St.JumpTicksLeft = St.JumpHeld ? (uint32_t)Rng.RangeInclusive(1, 3) : (uint32_t)Rng.RangeInclusive(5, 30);
	}
	St.Jump = St.JumpHeld ? 1 : 0;
	St.JumpTicksLeft--;

	if(St.HookTicksLeft == 0)
	{
		St.HookHeld = !St.HookHeld;
		if(St.HookHeld)
			St.HookTicksLeft = Rng.Chance(3, 20) ? (uint32_t)Rng.RangeInclusive(61, 120) : (uint32_t)Rng.RangeInclusive(1, 60);
		else
			St.HookTicksLeft = (uint32_t)Rng.RangeInclusive(3, 30);
	}
	St.Hook = St.HookHeld ? 1 : 0;
	St.HookTicksLeft--;

	if(St.AimNoiseTicksLeft == 0 || (St.Hook && Rng.Chance(1, 10)))
	{
		St.AimNoiseX = Rng.RangeInclusive(-30, 30);
		St.AimNoiseY = Rng.RangeInclusive(-30, 30);
		St.AimNoiseTicksLeft = (uint32_t)Rng.RangeInclusive(10, 50);
	}
	if(St.AimNoiseTicksLeft > 0)
		St.AimNoiseTicksLeft--;

	// Aim at the nearest other character (found above), plus noise -- "deliberate ... aim at
	// the nearest other character" per the task spec's acceptance criterion 6.
	int32_t TargetX = 1000, TargetY = 0; // fallback if alone (never (0,0) -- CNetObj_PlayerInput forbids aiming exactly at self).
	if(BestDistSq >= 0)
	{
		TargetX = (int32_t)NearestDx + St.AimNoiseX;
		TargetY = (int32_t)NearestDy + St.AimNoiseY;
	}
	if(TargetX == 0 && TargetY == 0)
		TargetX = 1000; // CNetObj_PlayerInput forbids aiming exactly at (0,0).

	// F7 (round-1 review): closed-loop block policy -- `m_WantedWeapon` is 1-BASED on the wire
	// (`HandleWeaponSwitch`, character.cpp: "if(m_LatestInput.m_WantedWeapon) WantedWeapon =
	// m_Input.m_WantedWeapon - 1;" -- 0 means "no explicit request", so the OLD `RangeInclusive
	// (0, 5)` generator request a weapon SWITCH only 5/6 of the time it rolled at all, and NEVER
	// actually requested hammer (WEAPON_HAMMER == 0 on the core side, but 1 on the wire) --
	// characters just stayed on the GUN they spawn with. Fixed: request hammer explicitly
	// (WantedWeapon = 1) whenever within hammer range of the nearest other character (~60
	// units, matching the task's "deliberate hammer hits ... near freeze edges"), so hammer
	// actually gets swung instead of gun being the default weapon for the entire corpus.
	constexpr int64_t HammerRangeSq = 90 * 90; // switch to hammer a bit before hitting range
	constexpr int64_t HammerHitRangeSq = 40 * 40; // CCharacter::GetProximityRadius()-scale melee range
	if(BestDistSq >= 0 && BestDistSq <= HammerRangeSq)
		St.WantedWeapon = 1; // hammer
	else if(Rng.Chance(1, 40))
		St.WantedWeapon = (int32_t)Rng.RangeInclusive(1, NUM_WEAPONS); // occasional variety

	// F7: `m_Fire` modeled as a real client's edge counter (see CharGenState's comment).
	// `FireWeapon()` only fires hammer on a PRESS EDGE (hammer is not in its `FullAuto` list,
	// so holding the button steady only ever fires once, on the edge into "held") -- a
	// per-tick independent coin flip while within actual melee range therefore generates many
	// press/release edges in quick succession (limited in practice by `m_ReloadTimer`'s own
	// cooldown, exactly like a player rapidly re-clicking), instead of one single press per
	// "close encounter" a steady level would have produced.
	bool WantFire = BestDistSq >= 0 && BestDistSq <= HammerHitRangeSq && Rng.Chance(1, 2);
	if(WantFire != St.FireHeld)
	{
		St.FireHeld = WantFire;
		St.FireCounter = (St.FireCounter + 1) & INPUT_STATE_MASK;
	}

	// F12 (round-2 review): generator policy for requesting a kill (the harness only APPLIES
	// the bit, via the real `OnKillNetMessage` -- see main()) -- after 3s (150 ticks at
	// SERVER_TICK_SPEED=50) continuously frozen, request one. `sv_kill_delay` (default 1s,
	// engine/shared/config_variables.h) means repeated requests while still frozen just get
	// ignored by `OnKillNetMessage` itself until it's ready, so setting the bit every tick
	// once past the threshold (rather than only once) is harmless and simpler than tracking
	// whether a request is still "pending".
	int32_t Kill = 0;
	if(WasFrozenLastTick)
	{
		if(++St.ContinuousFrozenTicks > 3 * (uint32_t)SERVER_TICK_SPEED)
			Kill = 1;
	}
	else
	{
		St.ContinuousFrozenTicks = 0;
	}

	ResolvedInput Out;
	Out.Direction = St.Direction;
	Out.TargetX = TargetX;
	Out.TargetY = TargetY;
	Out.Jump = St.Jump;
	Out.Fire = St.FireCounter;
	Out.Hook = St.Hook;
	Out.PlayerFlags = 0;
	Out.WantedWeapon = St.WantedWeapon;
	Out.NextWeapon = 0;
	Out.PrevWeapon = 0;
	Out.Kill = Kill;
	return Out;
}

} // namespace realmap_gen

// =============================================================================================
// trace-b v1 writer (docs/formats.md section 8) -- a NEW, separate format from Oracle A's
// "trace v1" (magic "TRC1"): Oracle A's format is locked in for task 1.3's parity work and this
// harness must not change it. This format's magic is "TRB1"; its first 28 state fields are
// byte-for-byte the same fields, in the same order, as Oracle A's `CharacterCoreState` (see
// docs/formats.md section 6.2) so a prefix-compare against an Oracle A trace is meaningful
// (used by --compare-oracle-a below), followed by the DDRace-level extension fields.
// =============================================================================================
namespace
{

struct CoreStateFields
{
	float PosX, PosY, VelX, VelY;
	float HookPosX, HookPosY, HookDirX, HookDirY, HookTeleBaseX, HookTeleBaseY;
	int32_t HookTick, HookState, HookedPlayer, ActiveWeapon, NewHook;
	int32_t Jumped, JumpedTotal, Jumps, Direction, Angle;
	int32_t TriggeredEvents, Colliding, LeftWall, MoveRestrictions;
	int32_t Solo, CollisionDisabled, EndlessHook, HookHitDisabled;

	void Write(ByteWriter &W) const
	{
		W.F32(PosX);
		W.F32(PosY);
		W.F32(VelX);
		W.F32(VelY);
		W.F32(HookPosX);
		W.F32(HookPosY);
		W.F32(HookDirX);
		W.F32(HookDirY);
		W.F32(HookTeleBaseX);
		W.F32(HookTeleBaseY);
		W.I32(HookTick);
		W.I32(HookState);
		W.I32(HookedPlayer);
		W.I32(ActiveWeapon);
		W.I32(NewHook);
		W.I32(Jumped);
		W.I32(JumpedTotal);
		W.I32(Jumps);
		W.I32(Direction);
		W.I32(Angle);
		W.I32(TriggeredEvents);
		W.I32(Colliding);
		W.I32(LeftWall);
		W.I32(MoveRestrictions);
		W.I32(Solo);
		W.I32(CollisionDisabled);
		W.I32(EndlessHook);
		W.I32(HookHitDisabled);
	}
};

struct DDRaceStateFields
{
	int32_t Alive = 0, DiedThisTick = 0, RespawnedThisTick = 0;
	int32_t FreezeTime = 0, IsInFreeze = 0, DeepFrozen = 0, LiveFrozen = 0, FrozenLastTick = 0;
	int32_t ReloadTimer = 0, AttackTick = 0, QueuedWeapon = -1, LastWeapon = 0;
	int32_t WeaponGotMask = 0;
	int32_t WeaponAmmo[NUM_WEAPONS] = {0, 0, 0, 0, 0, 0};
	int32_t WeaponAmmoRegenStart[NUM_WEAPONS] = {0, 0, 0, 0, 0, 0};
	int32_t NinjaActivationTick = 0, NinjaCurrentMoveTime = 0, NinjaOldVelAmount = 0;
	float NinjaActivationDirX = 0, NinjaActivationDirY = 0;
	int32_t TeleCheckpoint = 0;
	int32_t EndlessJump = 0, Jetpack = 0, Super = 0, Invincible = 0;
	int32_t HammerHitDisabled = 0, GrenadeHitDisabled = 0, LaserHitDisabled = 0, ShotgunHitDisabled = 0;
	int32_t HasTelegunGun = 0, HasTelegunGrenade = 0, HasTelegunLaser = 0;
	int32_t Team = 0, StrongWeakId = 0;
	// --- added in round-2 review fixes (F2, F6) ---
	int32_t FreezeStart = 0, FreezeEnd = 0; // F2: CCharacterCore::m_FreezeStart/m_FreezeEnd
	int32_t TuneZone = 0; // CCharacter::m_TuneZone
	int32_t NumInputs = 0; // CCharacter::m_NumInputs (fire/weapon-switch only act once > 1)
	int32_t LastRefillJumps = 0; // CCharacter::m_LastRefillJumps (0/1)
	int32_t DDRaceState = 0; // (int)CCharacter::m_DDRaceState (ERaceState)
	int32_t StartTime = 0; // CCharacter::m_StartTime
	int32_t DieTick = 0; // CPlayer::m_DieTick
	int32_t Spawning = 0; // CPlayer::m_Spawning (0/1) -- respawn-pending flag
	int32_t PreviousDieTick = 0; // F6 (round-2 review): CPlayer::m_PreviousDieTick -- earliest respawn tick

	void Write(ByteWriter &W) const
	{
		W.I32(Alive);
		W.I32(DiedThisTick);
		W.I32(RespawnedThisTick);
		W.I32(FreezeTime);
		W.I32(IsInFreeze);
		W.I32(DeepFrozen);
		W.I32(LiveFrozen);
		W.I32(FrozenLastTick);
		W.I32(ReloadTimer);
		W.I32(AttackTick);
		W.I32(QueuedWeapon);
		W.I32(LastWeapon);
		W.I32(WeaponGotMask);
		for(int i = 0; i < NUM_WEAPONS; i++)
			W.I32(WeaponAmmo[i]);
		for(int i = 0; i < NUM_WEAPONS; i++)
			W.I32(WeaponAmmoRegenStart[i]);
		W.I32(NinjaActivationTick);
		W.I32(NinjaCurrentMoveTime);
		W.I32(NinjaOldVelAmount);
		W.F32(NinjaActivationDirX);
		W.F32(NinjaActivationDirY);
		W.I32(TeleCheckpoint);
		W.I32(EndlessJump);
		W.I32(Jetpack);
		W.I32(Super);
		W.I32(Invincible);
		W.I32(HammerHitDisabled);
		W.I32(GrenadeHitDisabled);
		W.I32(LaserHitDisabled);
		W.I32(ShotgunHitDisabled);
		W.I32(HasTelegunGun);
		W.I32(HasTelegunGrenade);
		W.I32(HasTelegunLaser);
		W.I32(Team);
		W.I32(StrongWeakId);
		W.I32(FreezeStart);
		W.I32(FreezeEnd);
		W.I32(TuneZone);
		W.I32(NumInputs);
		W.I32(LastRefillJumps);
		W.I32(DDRaceState);
		W.I32(StartTime);
		W.I32(DieTick);
		W.I32(Spawning);
		W.I32(PreviousDieTick);
	}
};

// World entities relevant to physics that outlive a single character (F6): projectiles (gun/
// grenade) and lasers (shotgun/laser). Ninja's own "hit" pass has no persistent entity (handled
// inside HandleNinja() each tick, already covered by the character's own ninja fields).
// F6 (round-2 review): Kind values 2-6 are the 5 map-fixture classes that share the
// ENTTYPE_LASER slot with CLaser (see the dynamic_cast chain in main()). For 2/3/4/5
// (CDoor/CDragger/CDraggerBeam/CGun) only Kind/OwnerClientId(-1)/PosX/PosY are populated:
// every other field of those classes is declared before any explicit access-label in this
// DDNet build (C++ implicit-default-private), so `#define private public` -- a literal-token
// substitution -- cannot reach it without editing DDNet's own headers, which Oracle B never
// does. Kind 6 (CLight) is the exception: its *configuration* fields happen to sit after an
// explicit `public:` label in this build's light.h, so WeaponType/DirX/Extra are genuinely
// readable and reused as documented below (its dynamic sweep state, m_Core/m_Rotation, is
// implicit-private like the others and stays 0). See docs/formats.md section 8.2.
struct EntityRecord
{
	int32_t Kind = 0; // 0=CProjectile, 1=CLaser, 2=CDoor, 3=CDragger, 4=CDraggerBeam, 5=CGun, 6=CLight
	int32_t OwnerClientId = -1;
	int32_t WeaponType = 0; // CProjectile::m_Type / CLaser::m_Type (WEAPON_* constant); kind 6: CLight::m_Length
	float PosX = 0, PosY = 0;
	float DirX = 0, DirY = 0; // projectile: m_Direction; laser: m_Dir; kind 6: DirX=CLight::m_AngularSpeed
	int32_t StartTick = 0; // projectile: m_StartTick; laser: m_EvalTick
	int32_t Extra = 0; // projectile: m_LifeSpan; laser: m_Bounces; kind 6: CLight::m_Speed

	void Write(ByteWriter &W) const
	{
		W.I32(Kind);
		W.I32(OwnerClientId);
		W.I32(WeaponType);
		W.F32(PosX);
		W.F32(PosY);
		W.F32(DirX);
		W.F32(DirY);
		W.I32(StartTick);
		W.I32(Extra);
	}
};

// F14 (round-2 review): one entry per (dumped team, switch number). `switch_count`
// (`Collision()->m_HighestSwitchNumber`) and the LIST of teams dumped are both constant for an
// entire run (the run's characters' teams are fixed at spawn time), so they live ONCE in the
// trace-b file header (see main()'s header-writing code) rather than being repeated every tick
// -- only the actual per-tick status/end_tick/type/last_update_tick values live here.
struct SwitchEntry
{
	int32_t Status = 0, EndTick = 0, Type = 0, LastUpdateTick = 0;
	void Write(ByteWriter &W) const
	{
		W.I32(Status);
		W.I32(EndTick);
		W.I32(Type);
		W.I32(LastUpdateTick);
	}
};

struct GlobalTickFields
{
	int32_t GameTick = 0;
	// Flat [team_idx * switch_count + switch_idx], switch_idx 0 == switch number 1 -- team_idx
	// indexes into the file header's `switch_team_ids` list, in the same order.
	std::vector<SwitchEntry> Switches;
	std::vector<EntityRecord> Entities; // F6: live projectiles/lasers this tick

	void Write(ByteWriter &W) const
	{
		W.I32(GameTick);
		for(const SwitchEntry &S : Switches)
			S.Write(W);
		W.U32((uint32_t)Entities.size());
		for(const EntityRecord &E : Entities)
			E.Write(W);
	}
};

// Coverage counters (task spec acceptance criterion 6). Round-2 review (F3) fixed two bugs:
// hammer swings were being detected one tick late (missed entirely by the old `AttackTick ==
// GameTick` check -- `m_AttackTick` is stamped during `OnClientPredictedEarlyInput`, BEFORE
// `m_CurrentGameTick++`, so it is always `GameTick - 1` by the time state is dumped) and
// teleports were a crude position-jump heuristic that read 0 on maps where it happened anyway
// (`GetPureMapIndex`-based tile lookup on the character's previous position, matched against
// `CCollision::IsTeleport`/`IsEvilTeleport`/`IsCheckTeleport`/`IsCheckEvilTeleport`, is exact,
// not heuristic). Round-2 review round 3 (F13) refined this further: `IsTeleportHook` is
// deliberately EXCLUDED (it only ever redirects the HOOK, never the character standing on the
// tile) and a same-tick position-jump magnitude check confirms the teleport actually fired
// (guards a check-teleport tile that didn't -- e.g. no checkpoint reached yet). See the
// detection site itself (search "F13" below) and docs/formats.md section 9.7.
struct Coverage
{
	uint64_t CharacterTicks = 0;
	uint64_t FrozenTicks = 0, DeepFrozenTicks = 0, LiveFrozenTicks = 0;
	uint64_t FreezeEntries = 0, FreezeExits = 0;
	uint64_t SpeedupTicks = 0, TeleportTicks = 0;
	uint64_t HammerSwings = 0, HammerHits = 0;
	uint64_t HookGrabs = 0; // COREEVENT_HOOK_ATTACH_PLAYER edges (successful hook-on-player)
	uint64_t SwitchToggles = 0, TuneZoneTicks = 0;
	uint64_t StopperTicks = 0; // move_restrictions != 0
	uint64_t Died = 0, Respawned = 0;
	uint64_t SoloTicks = 0, JetpackTicks = 0, EndlessJumpTicks = 0, SuperTicks = 0;
	uint64_t CollisionDisabledTicks = 0, HookHitDisabledTicks = 0, NonZeroTeamTicks = 0;
};

} // namespace

int main(int argc, const char **argv)
{
	log_set_global_logger_default();

	// -------------------------------------------------------------------------------------
	// Argument parsing
	// -------------------------------------------------------------------------------------
	std::string StorageDir, OutPath, RawmapPath, ScenarioPath, RealMapPath, CompareOracleAPath, CoverageOutPath, EmitScenarioV3Path;
	std::vector<std::string> CfgFiles;
	std::vector<std::pair<uint32_t, int32_t>> TeamOverrides;
	std::string GeneratorName;
	bool HasSeed = false;
	uint64_t Seed = 0;
	bool HasTicks = false;
	uint32_t Ticks = 0;
	bool HasChars = false;
	uint32_t Chars = 0;

	auto NeedArg = [&](int &i) -> std::string {
		if(i + 1 >= argc)
			Fail(std::string("missing value for ") + argv[i]);
		return argv[++i];
	};

	for(int i = 1; i < argc; i++)
	{
		std::string A = argv[i];
		if(A == "--storage-dir")
			StorageDir = NeedArg(i);
		else if(A == "--out")
			OutPath = NeedArg(i);
		else if(A == "--rawmap")
			RawmapPath = NeedArg(i);
		else if(A == "--scenario")
			ScenarioPath = NeedArg(i);
		else if(A == "--real-map")
			RealMapPath = NeedArg(i);
		else if(A == "--seed")
		{
			Seed = strtoull(NeedArg(i).c_str(), nullptr, 10);
			HasSeed = true;
		}
		else if(A == "--ticks")
		{
			Ticks = (uint32_t)strtoul(NeedArg(i).c_str(), nullptr, 10);
			HasTicks = true;
		}
		else if(A == "--chars")
		{
			Chars = (uint32_t)strtoul(NeedArg(i).c_str(), nullptr, 10);
			HasChars = true;
		}
		else if(A == "--cfg")
			CfgFiles.push_back(NeedArg(i));
		else if(A == "--team")
		{
			std::string Spec = NeedArg(i);
			size_t Eq = Spec.find('=');
			if(Eq == std::string::npos)
				Fail("--team expects id=team");
			TeamOverrides.emplace_back((uint32_t)strtoul(Spec.substr(0, Eq).c_str(), nullptr, 10), (int32_t)strtol(Spec.substr(Eq + 1).c_str(), nullptr, 10));
		}
		else if(A == "--generator")
			GeneratorName = NeedArg(i);
		else if(A == "--compare-oracle-a")
			CompareOracleAPath = NeedArg(i);
		else if(A == "--coverage-out")
			CoverageOutPath = NeedArg(i);
		else if(A == "--emit-scenario-v3")
			EmitScenarioV3Path = NeedArg(i);
		else
			Fail("unknown argument: " + A);
	}

	if(StorageDir.empty())
		Fail("--storage-dir is required");
	if(OutPath.empty())
		Fail("--out is required");
	if(!HasSeed)
		Fail("--seed is required (deterministic PRNG reseed -- see docs/formats.md section 9.2)");
	bool RawmapMode = !RawmapPath.empty();
	bool RealMapMode = !RealMapPath.empty();
	if(RawmapMode == RealMapMode)
		Fail("exactly one of --rawmap+--scenario or --real-map is required");
	if(RawmapMode && ScenarioPath.empty())
		Fail("--rawmap requires --scenario");
	if(RealMapMode && !(HasTicks && HasChars))
		Fail("--real-map requires --ticks and --chars");

	(void)fs_makedir(StorageDir.c_str());
	std::string MapsDir = StorageDir + "/maps";
	(void)fs_makedir(MapsDir.c_str());

	// -------------------------------------------------------------------------------------
	// Resolve map input (docs/formats.md section 8/9): either raw2map a rawmap into the
	// storage dir, or place/symlink the given real .map file there, under a name LoadMap() can
	// address. Either way, the harness's OWN sha256 of the actual bytes it hands to the real
	// LoadMap()/CDataFileReader path (not merely the scenario's declared one) goes into the
	// trace's metadata, so the metadata always reflects what was *actually* simulated.
	// -------------------------------------------------------------------------------------
	std::string MapName = "oracle_b_map";
	std::array<unsigned char, 32> ActualMapSha256{};
	Scenario S;
	std::vector<unsigned char> ScenarioBytes;
	std::array<unsigned char, 32> ScenarioSha256{};

	if(RawmapMode)
	{
		std::vector<unsigned char> RawmapBytes = ReadFile(RawmapPath);
		auto RawmapDigest = oracle_sha256::Digest(RawmapBytes);
		RawMap Map = ParseRawMap(RawmapBytes);

		ScenarioBytes = ReadFile(ScenarioPath);
		ScenarioSha256 = oracle_sha256::Digest(ScenarioBytes);
		S = ParseScenario(ScenarioBytes);

		if(memcmp(RawmapDigest.data(), S.MapSha256.data(), 32) != 0)
			Fail("rawmap sha256 does not match scenario's declared map_sha256");

		// Written directly under the storage dir's maps/ folder below via WriteMapFromRawMap,
		// which is called after storage exists (see the CServer bootstrap section).
		ActualMapSha256 = RawmapDigest; // overwritten below with the .map file's own sha256 once written
	}
	else
	{
		// Copy the real map into storage/maps/<MapName>.map so IStorage (rooted at StorageDir)
		// can find it under a name we control, without ever writing outside StorageDir.
		std::vector<unsigned char> MapBytes = ReadFile(RealMapPath);
		ActualMapSha256 = oracle_sha256::Digest(MapBytes);
		WriteFile(MapsDir + "/" + MapName + ".map", MapBytes);
	}

	// -------------------------------------------------------------------------------------
	// Kernel/engine/console/config/http/antibot/game-server bootstrap -- mirrors DDNet's own
	// src/test/gameworld_test.cpp `GameWorld::GameWorld()` constructor (see docs/formats.md
	// section 8.1 for the file:line derivation), except:
	//   * storage is rooted at our own `--storage-dir` (via CreateTempStorage, the same factory
	//     gameworld_test.cpp uses through CTestInfo::CreateTestStorage) instead of a name gtest
	//     derives from the current test's name -- this harness has no gtest;
	//   * console commands (server config vars / tune commands, acceptance criterion 2) are
	//     executed right after RegisterCommands(), before LoadMap/OnInit -- the same point
	//     src/engine/server/main.cpp executes autoexec.cfg/command-line arguments at (see
	//     docs/formats.md section 8.2's file:line citations);
	//   * we never call m_NetServer.Open()/Run() (no socket), m_Econ.Init() (no socket), or
	//     m_pRegister->Update() (no master registration) -- this harness never opens a network
	//     socket or registers anywhere, matching the task spec's constraints;
	//   * `CreatePlayer`+`ForceSpawn` are called directly (not `OnClientConnected`/
	//     `OnClientEnter`, which try to send welcome-message/vote/server-info network packets)
	//     -- the exact same pattern gameworld_test.cpp's own `BasicTick`/`CharacterEmote` tests
	//     use.
	// -------------------------------------------------------------------------------------
	CConfig ConfigBackup = g_Config;

	CServer *pServer = CreateServer();
	std::unique_ptr<IKernel> pKernel(IKernel::Create());
	pKernel->RegisterInterface(pServer);

	IEngine *pEngine = CreateTestEngine(GAME_NAME);
	pKernel->RegisterInterface(pEngine);

	const char *apArgs[] = {argv[0]};
	std::unique_ptr<IStorage> pStorage = CreateTempStorage(StorageDir.c_str(), 1, apArgs);
	if(!pStorage)
		Fail("failed to create storage rooted at " + StorageDir);
	pKernel->RegisterInterface(pStorage.get(), false);

	// `RegisterInterface`'s default `Destroy=true` means the KERNEL takes ownership and deletes
	// this on its own destruction -- `.release()` (matching gameworld_test.cpp's own
	// `CreateConsole(...).release()`) hands it a raw pointer instead of keeping a `unique_ptr`
	// on this side too, which would otherwise free the same object twice.
	IConsole *pConsole = CreateConsole(CFGFLAG_SERVER | CFGFLAG_ECON).release();
	pKernel->RegisterInterface(pConsole);

	IConfigManager *pConfigManager = CreateConfigManager();
	pKernel->RegisterInterface(pConfigManager);

	IEngineHttp *pEngineHttp = CreateEngineHttp();
	pKernel->RegisterInterface(pEngineHttp);
	pKernel->RegisterInterface(static_cast<IHttp *>(pEngineHttp), false);

	IEngineAntibot *pEngineAntibot = CreateEngineAntibot();
	pKernel->RegisterInterface(pEngineAntibot);
	pKernel->RegisterInterface(static_cast<IAntibot *>(pEngineAntibot), false);

	IGameServer *pGameServerIface = CreateGameServer();
	pKernel->RegisterInterface(pGameServerIface);
	CGameContext *pGameServer = (CGameContext *)pGameServerIface;

	pEngine->Init();
	pConsole->Init();
	pConfigManager->Init();

	pServer->RegisterCommands();

	// raw2map now that storage/console exist (needs IStorage; harmless that console commands
	// haven't executed yet -- map writing doesn't depend on config).
	if(RawmapMode)
	{
		std::vector<unsigned char> RawmapBytes = ReadFile(RawmapPath);
		RawMap Map = ParseRawMap(RawmapBytes);
		if(!WriteMapFromRawMap(Map, pStorage.get(), MapName))
			Fail("raw2map: failed to write " + MapsDir + "/" + MapName + ".map");
		std::vector<unsigned char> WrittenBytes = ReadFile(MapsDir + "/" + MapName + ".map");
		ActualMapSha256 = oracle_sha256::Digest(WrittenBytes);
	}

	// Console commands (acceptance criterion 2): server config variables and tune commands,
	// executed before the first tick, in the same relative position as a real server's
	// autoexec.cfg (main.cpp:174-186 in the fetched tree).
	for(const std::string &CfgFile : CfgFiles)
	{
		if(!pConsole->ExecuteFile(CfgFile.c_str(), IConsole::CLIENT_ID_UNSPECIFIED, true, IStorage::TYPE_ALL_OR_ABSOLUTE))
			Fail("failed to execute cfg file " + CfgFile);
	}
	// F5: scenario v3's own embedded cfg lines (self-contained replay -- see WriteScenarioV3)
	// run at the same two points a `--cfg` FILE would (this is the pre-init pass; the F1-fix
	// post-init re-run below repeats them too).
	if(RawmapMode)
		for(const std::string &Line : S.CfgLines)
			pConsole->ExecuteLine(Line.c_str(), IConsole::CLIENT_ID_UNSPECIFIED);
	if(RawmapMode && S.NoWeakHook)
		pConsole->ExecuteLine("sv_no_weak_hook 1", IConsole::CLIENT_ID_UNSPECIFIED);

	{
		int Size = pGameServer->PersistentClientDataSize();
		for(auto &Client : pServer->m_aClients)
		{
			Client.m_HasPersistentData = false;
			Client.m_pPersistentData = malloc(Size);
		}
	}
	pServer->m_pPersistentData = malloc(pGameServer->PersistentDataSize());

	pServer->m_RunServer = CServer::RUNNING;
	if(!pServer->LoadMap(MapName.c_str()))
		Fail("LoadMap(" + MapName + ") failed -- see stderr above for the engine's own error");

	pServer->m_AuthManager.Init();
	pServer->Antibot()->Init();
	pGameServer->OnInit(nullptr);

	// PRNG reseed (acceptance criterion 2 -- "fixed PRNG seed(s) replacing any secure-random
	// seeding that affects gameplay"). The ONLY such place in the server/game tick path is
	// `CGameContext::m_Prng` (gamecontext.cpp's OnInit calls `secure_random_fill` then
	// `m_Prng.Seed`; `m_World.m_Core.m_pPrng = &m_Prng` makes it the core's `CWorldCore::
	// RandomOr0`, used solely for picking a random TELEOUT among several with the same number
	// when a hook flies through a TELEINHOOK tile -- game/gamecore.cpp:399 and
	// game/server/entities/character.cpp:2021/2036/2063/2100 for TELEIN/TELEINEVIL/TELECHECK*).
	// `CScore`'s own separate `m_Prng` (game/server/score.cpp) only ever feeds a chat-message
	// word list on race finish -- irrelevant to any traced field, and never reached by this
	// harness anyway (no database, no chat). See docs/formats.md section 9.2.
	//
	// `CGameContext::m_Prng` itself sits in that class's *implicit* default-private region (no
	// `private:` keyword precedes it -- it is simply before the class's first `public:` label),
	// so the `#define private public` trick above (which only rewrites the literal `private`/
	// `protected` keywords) cannot reach it. Reseeding through the already-*public*
	// `CWorldCore::m_pPrng` pointer (`m_World.m_Core.m_pPrng`, gamecore.h) -- which `OnInit()`
	// just pointed at this exact same `CPrng` instance -- reaches the identical object without
	// needing any private-access workaround at all.
	{
		uint64_t aSeed[2] = {Seed, Seed ^ 0x9E3779B97F4A7C15ULL};
		pGameServer->m_World.m_Core.m_pPrng->Seed(aSeed);
	}

	// Apply tuning overrides directly (acceptance criterion 2 and docs/formats.md section 2 --
	// the same rationale as Oracle A's oracle_core.cpp: `CTuningParams::Set` multiplies by 100
	// again internally, so poking `NetworkArray()` directly is the only way to apply an exact
	// `value_x100` integer without a double float round-trip). Applied to `GlobalTuning()`
	// (tune zone 0) -- the only zone a spawn position with no TILE_TUNE tile underneath it
	// resolves to, i.e. every character in every one of this harness's scenarios/generated runs.
	if(RawmapMode)
	{
		for(const auto &O : S.TuningOverrides)
		{
			int Index = -1;
			for(int i = 0; i < CTuningParams::Num(); i++)
				if(str_comp_nocase(CTuningParams::Name(i), O.Name.c_str()) == 0)
				{
					Index = i;
					break;
				}
			if(Index < 0)
				Fail("unknown tuning parameter '" + O.Name + "'");
			pGameServer->GlobalTuning()->NetworkArray()[Index] = O.ValueX100;
		}
	}

	// F1 fix (round-1 review): `--cfg` was only ever executed ONCE, before `LoadMap`/`OnInit` --
	// correct for `sv_*` vars that must precede map load (`sv_solo_server`'s own doc comment:
	// "has to be set before loading the map"), but `CGameContext::OnInit` (gamecontext.cpp:
	// ~4110-4119) unconditionally resets `TuningList()[i] = CTuningParams::DEFAULT` for EVERY
	// tune zone, INCLUDING zone 0, as part of its own startup -- so a `tune`/`tune_zone` command
	// in that first pass was silently discarded by `OnInit()` itself, no matter what it set.
	// Confirmed by reproduction: a `--cfg` with `tune gravity 0` produced a BYTE-IDENTICAL trace
	// to a run with no `--cfg` at all (`vel_y` still -12.19/-13.00/-13.73/... every tick) even
	// though the console printed "tuning: gravity changed to 0.00" -- the value took effect on
	// `GlobalTuning()` for a moment and was then overwritten by `OnInit()`'s own reset, which
	// runs unconditionally regardless of `sv_tune_reset` (that flag only gates a SEPARATE,
	// later `ResetTuning()` call -- gamecontext.cpp's per-zone loop above it is unconditional).
	//
	// Fix: re-execute every `--cfg` file AGAIN here, after `OnInit()` and after the scenario's
	// own tuning overrides (which use the direct `NetworkArray()` poke, unaffected by this bug,
	// and should still be the last word if a `--cfg` and the scenario disagree on the same
	// parameter -- so scenario overrides are applied first, cfg re-run second). Re-running the
	// pre-init pass a second time is harmless: `sv_*` config vars are simple idempotent
	// assignments (setting `sv_solo_server 1` again after `OnInit()` already ran does not
	// re-trigger `OnInit()`'s one-time `if(g_Config.m_SvSoloServer)` branch a second time --
	// that code already ran once, during the first pass, when it mattered); `tune`/`tune_zone`
	// are exactly the commands this second pass exists to make stick. `--cfg` is documented
	// (README/formats.md) as being for config/tuning commands specifically, executed twice, for
	// this reason -- not for one-shot administrative commands like `say`.
	for(const std::string &CfgFile : CfgFiles)
	{
		if(!pConsole->ExecuteFile(CfgFile.c_str(), IConsole::CLIENT_ID_UNSPECIFIED, true, IStorage::TYPE_ALL_OR_ABSOLUTE))
			Fail("failed to re-execute cfg file " + CfgFile + " after OnInit()");
	}
	if(RawmapMode)
		for(const std::string &Line : S.CfgLines)
			pConsole->ExecuteLine(Line.c_str(), IConsole::CLIENT_ID_UNSPECIFIED);

	// -------------------------------------------------------------------------------------
	// Characters: spawn positions + ids, either from the scenario (rawmap mode) or from the
	// harness-side real-map generator (real-map mode).
	// -------------------------------------------------------------------------------------
	std::vector<uint32_t> CharIds;
	std::vector<std::pair<int32_t, int32_t>> SpawnPositions;
	uint32_t NumChars = 0;
	std::vector<realmap_gen::CharGenState> GenState;
	std::unique_ptr<SplitMix64> pGenRng;

	if(RawmapMode)
	{
		NumChars = (uint32_t)S.Characters.size();
		if(NumChars == 0)
			Fail("scenario has no characters");
		// F10 (round-1 review): `--ticks` used to be silently accepted and, if it exceeded the
		// scenario's own recorded tick count, ran straight into an out-of-bounds
		// `S.Inputs[Tick]` access in the tick loop below (`_GLIBCXX_ASSERTIONS` aborts the
		// process rather than reading garbage, but the failure mode was an unexplained abort,
		// not a clear error message). `--chars` is a real-map-mode-only flag -- the scenario
		// file itself is authoritative for character count in this mode -- so a `--chars` that
        // does not match is now a clear error instead of being silently ignored.
		if(!HasTicks)
			Ticks = (uint32_t)S.Inputs.size();
		else if(Ticks > S.Inputs.size())
			Fail("--ticks " + std::to_string(Ticks) + " exceeds the scenario's own tick count (" +
				std::to_string(S.Inputs.size()) + "); omit --ticks to use the scenario's length, or pass a smaller value");
		if(HasChars && Chars != NumChars)
			Fail("--chars " + std::to_string(Chars) + " does not match the scenario's character count (" +
				std::to_string(NumChars) + "); --chars only applies to --real-map mode, omit it here");
		// F5: a v3 scenario embeds the seed its inputs were generated under -- replaying it
		// with a DIFFERENT --seed would reseed the PRNG differently (docs/formats.md section
		// 9.2) and, on a map with multiple TELEOUTs sharing a number, could silently diverge
		// from the original run despite replaying the exact same recorded inputs. Enforced here
		// rather than silently trusting the caller.
		if(S.Version >= 3 && S.EmbeddedSeed != Seed)
			Fail("--seed " + std::to_string(Seed) + " does not match this v3 scenario's embedded seed (" +
				std::to_string(S.EmbeddedSeed) + "); pass --seed " + std::to_string(S.EmbeddedSeed) + " to replay it faithfully");
		for(auto &C : S.Characters)
		{
			CharIds.push_back(C.Id);
			SpawnPositions.emplace_back(C.SpawnX, C.SpawnY);
		}
	}
	else
	{
		NumChars = Chars;
		if(NumChars == 0 || NumChars > 4)
			Fail("--chars must be in 1..=4 (task spec: 2-4 characters per real-map scenario)");
		pGenRng = std::make_unique<SplitMix64>(Seed);
		auto Spawns = realmap_gen::ChooseSpawns(pGameServer->Collision(), *pGenRng, NumChars);
		for(uint32_t i = 0; i < NumChars; i++)
		{
			CharIds.push_back(i);
			SpawnPositions.emplace_back(Spawns[i].X * 32 + 16, Spawns[i].Y * 32 + 16);
		}
		GenState.resize(NumChars);
	}

	for(uint32_t Slot = 0; Slot < NumChars; Slot++)
	{
		if(CharIds[Slot] >= (uint32_t)MAX_CLIENTS)
			Fail("character id out of range");
		for(uint32_t Other = 0; Other < Slot; Other++)
			if(CharIds[Other] == CharIds[Slot])
				Fail("duplicate character id");
	}

	std::vector<int32_t> EffectiveTeam(NumChars, 0);
	for(uint32_t Slot = 0; Slot < NumChars; Slot++)
	{
		int Id = (int)CharIds[Slot];
		pGameServer->CreatePlayer(Id, TEAM_GAME, /*Afk=*/false, /*LastWhisperTo=*/-1);
		pServer->m_aClients[Id].m_State = CServer::CClient::STATE_INGAME;
		CPlayer *pPlayer = pGameServer->m_apPlayers[Id];
		pPlayer->ForceSpawn(vec2((float)SpawnPositions[Slot].first, (float)SpawnPositions[Slot].second));
		// F5: a v3 scenario's own per-character team is the baseline; an explicit --team on
		// the replay command line (checked second, below) still wins if given, matching how a
		// --cfg file's tune commands can still be overridden by a later one in formats.md's
		// documented precedence.
		if(RawmapMode && S.Version >= 3 && S.Characters[Slot].Team != 0)
		{
			pGameServer->m_pController->Teams().SetForceCharacterTeam(Id, S.Characters[Slot].Team);
			EffectiveTeam[Slot] = S.Characters[Slot].Team;
		}
		for(const auto &TO : TeamOverrides)
			if((int)TO.first == Id)
			{
				pGameServer->m_pController->Teams().SetForceCharacterTeam(Id, TO.second);
				EffectiveTeam[Slot] = TO.second;
			}

		// Coverage addition (round-2 review, orchestrator addition to F7): "give some characters
		// shotgun/grenade/laser variety ... so those paths get real coverage" -- vanilla DDRace
		// spawn only ever grants hammer+gun (IGameController::OnCharacterSpawn,
		// gamecontroller.cpp), so without this every non-hammer shot in the corpus depended
		// entirely on incidental weapon-pickup tiles existing on a given real map. `GiveWeapon`
		// is the same real, public, unmodified `CCharacter` API a `CPickup` uses on walkover
		// (entities/pickup.cpp) -- calling it here is not new/simulated game behaviour, just the
		// harness choosing a starting loadout the way a custom gametype's `OnCharacterSpawn`
		// would. Kept REPLAY-SAFE (matches this file's own F12 lesson about undeclared harness
		// side effects breaking byte-identical replay): the decision is a pure function of
		// `Seed` and this character's `Id` alone, NOT of the per-tick generator PRNG stream, and
		// this whole loop already runs identically for fresh generation (--real-map) and for
		// replaying a previously recorded --scenario/--rawmap pair -- so a replay of any trace
		// this produces reaches the exact same decision from the exact same (Seed, Id), with no
        // new data needed in the scenario file.
		uint64_t WeaponRoll = SplitMix64((uint64_t)Seed * 1000003ULL + 97ULL * (uint64_t)Id).NextU64() & 3;
		if(CCharacter *pNewChar = pPlayer->GetCharacter())
		{
			if(WeaponRoll == 0)
				pNewChar->GiveWeapon(WEAPON_SHOTGUN);
			else if(WeaponRoll == 1)
				pNewChar->GiveWeapon(WEAPON_GRENADE);
			else if(WeaponRoll == 2)
				pNewChar->GiveWeapon(WEAPON_LASER);
			// WeaponRoll == 3: keep the default hammer+gun-only loadout.
		}
	}

	// F14 (round-2 review): switchers are dumped for every DDRace team actually present among
	// this run's characters, not just team 0 (TEAM_FLOCK) -- always includes 0 even if every
	// character was moved to another team, since 0 remains a meaningful reference point (e.g.
	// a map-fixture entity like a turret/dragger with no explicit team still reads as 0).
	std::vector<int32_t> TeamsDumped;
	TeamsDumped.push_back(TEAM_FLOCK);
	for(int32_t T : EffectiveTeam)
		if(std::find(TeamsDumped.begin(), TeamsDumped.end(), T) == TeamsDumped.end())
			TeamsDumped.push_back(T);
	const int32_t HighestSwitchNumber = std::max(0, pGameServer->Collision()->m_HighestSwitchNumber);

	// -------------------------------------------------------------------------------------
	// Tick loop (acceptance criterion 3): reproduces engine/server/server.cpp's `Run()` inner
	// loop body exactly (lines ~3547-3597 in the fetched tree; see docs/formats.md section 8.3
	// for the full file:line derivation) -- `OnPreTickTeehistorian` -> for each in-game client,
	// `OnClientPredictedEarlyInput` -> `m_CurrentGameTick++` -> for each in-game client,
	// `OnClientPredictedInput` -> `OnTick()`. The real loop walks `c` over `0..MAX_CLIENTS`
	// (not our own character list order) and skips any client whose state isn't
	// `STATE_INGAME` -- reproduced identically below.
	//
	// Every character in this harness has fresh, non-null input on every single tick from
	// tick 0 onward (never the "no packet arrived this tick, reuse the last one" branch),
	// mirroring an idealized zero-jitter client: `CGameContext::OnClientPredictedEarlyInput`/
	// `OnClientPredictedInput` (gamecontext.cpp:1585-1635) only take the "reuse
	// m_aLastPlayerInput" branch when their `pInput` argument is `nullptr`, which never happens
	// here. `m_NumInputs` (`CPlayer::OnPredictedInput`, player.cpp:642) therefore increments
	// every tick for every character, and `PlayerFlags` (`CPlayer::OnPredictedEarlyInput`,
	// player.cpp:680, `m_PlayerFlags = pNewInput->m_PlayerFlags`) tracks whatever this harness's
	// scenario/generator input sets it to every tick (0 unless a scenario explicitly sets
	// `player_flags`, which none of the current recipes/generator do).
	// -------------------------------------------------------------------------------------
	std::vector<std::pair<int32_t, int32_t>> PrevPositions = SpawnPositions;
	std::vector<bool> WasAlive(NumChars, true);
	std::vector<bool> WasInFreeze(NumChars, false);
	std::vector<int32_t> PrevAttackTick(NumChars, std::numeric_limits<int32_t>::min());
	// Same "stamped during the pre-increment phase" situation as `PrevAttackTick` above, for
	// `CPlayer::m_DieTick`: `OnKillNetMessage`'s `KillCharacter()`/`Die()` runs from this file's
	// kill-application block, BEFORE `m_CurrentGameTick++` -- so `m_DieTick` is stamped to the
	// OLD (pre-increment) tick number, one behind the `GameTick` this tick's row gets dumped
	// under, and comparing it against `Glob.GameTick` directly would never match. Tracking
	// CHANGES against the last-recorded value (exactly like `PrevAttackTick`) sidesteps needing
	// to know the exact off-by-one and catches a same-tick kill-then-respawn (where the `Alive`
	// flag itself never visibly flips) that a pure before/after `Alive` comparison would miss.
	std::vector<int32_t> PrevDieTick(NumChars, std::numeric_limits<int32_t>::min());
	std::vector<int32_t> PrevSwitchStatus; // empty on tick 0 -- see the switch-toggle check below

	std::vector<std::vector<CoreStateFields>> AllCore(Ticks, std::vector<CoreStateFields>(NumChars));
	std::vector<std::vector<DDRaceStateFields>> AllDDRace(Ticks, std::vector<DDRaceStateFields>(NumChars));
	std::vector<std::vector<ResolvedInput>> AllInputs(Ticks, std::vector<ResolvedInput>(NumChars));
	std::vector<GlobalTickFields> AllGlobal(Ticks);
	Coverage Cov;

	auto SlotForClientId = [&](int ClientId) -> int {
		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
			if((int)CharIds[Slot] == ClientId)
				return (int)Slot;
		return -1;
	};

	auto Start = std::chrono::steady_clock::now();
	for(uint32_t Tick = 0; Tick < Ticks; Tick++)
	{
		std::vector<CNetObj_PlayerInput> NetInputs(NumChars);
		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		{
			ResolvedInput In;
			if(RawmapMode)
				In = ResolveInput(S.Inputs[Tick][Slot], Slot, PrevPositions);
			else
			{
				bool WasFrozen = Tick > 0 && AllDDRace[Tick - 1][Slot].IsInFreeze != 0;
				In = realmap_gen::StepCharGen(GenState[Slot], *pGenRng, Slot, PrevPositions, WasFrozen);
			}
			AllInputs[Tick][Slot] = In;
			CNetObj_PlayerInput &NetIn = NetInputs[Slot];
			NetIn.m_Direction = In.Direction;
			NetIn.m_TargetX = In.TargetX;
			NetIn.m_TargetY = In.TargetY;
			NetIn.m_Jump = In.Jump;
			NetIn.m_Fire = In.Fire;
			NetIn.m_Hook = In.Hook;
			NetIn.m_PlayerFlags = In.PlayerFlags;
			NetIn.m_WantedWeapon = In.WantedWeapon;
			NetIn.m_NextWeapon = In.NextWeapon;
			NetIn.m_PrevWeapon = In.PrevWeapon;
		}

		// F12 (round-2 review): a recorded "kill" request is applied exactly like a real
		// client's `CNetMsg_Cl_Kill` message -- `CGameContext::OnKillNetMessage`
		// (gamecontext.cpp) itself, not a bespoke harness action. A real server processes
		// client messages as network packets arrive, continuously, BETWEEN the tick-loop's
		// iterations (`server.cpp`'s `PumpNetwork()`, outside the `while(LastTime > ...)` tick
		// loop) -- so by the time a given tick's `OnPreTickTeehistorian()`/input phase runs,
		// any kill message for THIS tick has already taken effect. Calling it here, right
		// before that phase, reproduces exactly that ordering. `OnKillNetMessage` itself
		// enforces `sv_kill_delay` (`m_LastKill`) and kill-protection exactly as it would for a
		// real client -- this harness does not reimplement or bypass either.
		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		{
			if(!AllInputs[Tick][Slot].Kill)
				continue;
			CNetMsg_Cl_Kill KillMsg{};
			pGameServer->OnKillNetMessage(&KillMsg, (int)CharIds[Slot]);
		}

		pGameServer->OnPreTickTeehistorian();
		for(int c = 0; c < MAX_CLIENTS; c++)
		{
			if(pServer->m_aClients[c].m_State != CServer::CClient::STATE_INGAME)
				continue;
			int Slot = SlotForClientId(c);
			pGameServer->OnClientPredictedEarlyInput(c, Slot >= 0 ? &NetInputs[Slot] : nullptr);
		}

		pServer->m_CurrentGameTick++;

		for(int c = 0; c < MAX_CLIENTS; c++)
		{
			if(pServer->m_aClients[c].m_State != CServer::CClient::STATE_INGAME)
				continue;
			int Slot = SlotForClientId(c);
			pGameServer->OnClientPredictedInput(c, Slot >= 0 ? &NetInputs[Slot] : nullptr);
		}

		pGameServer->OnTick();

		// --- Dump ---------------------------------------------------------------------------
		GlobalTickFields &Glob = AllGlobal[Tick];
		Glob.GameTick = pServer->Tick();
		{
			// F14 (round-2 review): dump switchers for every DDRace team actually present among
			// this run's characters (`TeamsDumped`, computed once after character creation --
			// see below), not just team 0 -- a run that puts everyone on team 3 (this corpus's
			// own `--team` cfg-variation slice, see bulk_run_server.sh) previously had NO way to
			// see its own doors' state in the trace at all.
			auto &Switchers = pGameServer->Switchers();
			Glob.Switches.resize(TeamsDumped.size() * HighestSwitchNumber);
			for(size_t TeamIdx = 0; TeamIdx < TeamsDumped.size(); TeamIdx++)
			{
				int Team = TeamsDumped[TeamIdx];
				for(size_t i = 1; i < Switchers.size() && i <= (size_t)HighestSwitchNumber; i++)
				{
					SwitchEntry &E = Glob.Switches[TeamIdx * HighestSwitchNumber + (i - 1)];
					E.Status = Switchers[i].m_aStatus[Team] ? 1 : 0;
					E.EndTick = Switchers[i].m_aEndTick[Team];
					E.Type = Switchers[i].m_aType[Team];
					E.LastUpdateTick = Switchers[i].m_aLastUpdateTick[Team];
				}
			}
			// F3: exact switch-toggle count (status changed since the previous tick, any
			// dumped team) -- `PrevSwitchStatus` starts empty, so tick 0 never counts a
			// spurious toggle.
			for(size_t i = 0; i < Glob.Switches.size(); i++)
				if(i < PrevSwitchStatus.size() && PrevSwitchStatus[i] != Glob.Switches[i].Status)
					Cov.SwitchToggles++;
			PrevSwitchStatus.clear();
			for(const SwitchEntry &E : Glob.Switches)
				PrevSwitchStatus.push_back(E.Status);
		}
		// F6: world entities that outlive a single tick -- live projectiles (gun/grenade) and
		// lasers (shotgun/laser), walked via CGameWorld's per-type linked list.
		{
			// SAFETY BUG FIX (found while verifying the corpus, post round-2-review): `CEntity`'s
			// `ENTTYPE_LASER` slot (`gameworld.h`) is NOT exclusive to `CLaser` -- SIX different
			// classes register under the very same tag: `CLaser`, `CLight`, `CDraggerBeam`,
			// `CDoor`, `CDragger`, `CGun` (grep across `game/server/entities/*.cpp` confirms all
			// six construct their `CEntity` base with `ENTTYPE_LASER`). An unconditional
			// `static_cast<CLaser*>` on every entity `FindFirst(ENTTYPE_LASER)` returns therefore
			// read `CLaser`-shaped fields out of a `CDoor`/`CGun`/... object whenever a map has
			// any of those (map-fixture entities) -- garbage values, and NOT deterministic in
			// the sense this schema needs: reproduced on `BlmapChill` (203 such entities) -- two
			// separate --real-map runs of the exact same seed produced byte-identical CHARACTER
			// rows (physics is fine) but different garbage in these misread entity records.
			// `dynamic_cast` (RTTI is enabled -- no `-fno-rtti` in this build, confirmed in the
			// compile command) safely identifies the real runtime type instead of assuming.
			//
			// F6 (round-2 review): the 5 map-fixture classes are now dumped too (BlmapChill has
			// turrets/draggers) -- kinds 2-6, using only PUBLIC-shape-safe fields each class
			// actually has (checked field-by-field against game/server/entities/{door,dragger,
			// dragger_beam,gun,light}.h): position always meaningful; `dir`/`extra` reused
			// per-kind as documented in docs/formats.md section 8.2 (e.g. dragger's `m_Strength`
			// as extra*1000 fixed-point); `owner_client_id`/`weapon_type` are `-1`/`0` ("not
			// applicable") except for CDraggerBeam, where `owner_client_id` is genuinely
			// meaningful (`m_ForClientId`, the player being dragged).
			for(CEntity *pEnt = pGameServer->m_World.FindFirst(CGameWorld::ENTTYPE_PROJECTILE); pEnt; pEnt = pEnt->m_pNextTypeEntity)
			{
				auto *pProj = dynamic_cast<CProjectile *>(pEnt);
				if(!pProj)
					continue;
				EntityRecord R;
				R.Kind = 0;
				R.OwnerClientId = pProj->m_Owner;
				R.WeaponType = pProj->m_Type;
				R.PosX = pProj->m_Pos.x;
				R.PosY = pProj->m_Pos.y;
				R.DirX = pProj->m_Direction.x;
				R.DirY = pProj->m_Direction.y;
				R.StartTick = pProj->m_StartTick;
				R.Extra = pProj->m_LifeSpan;
				Glob.Entities.push_back(R);
			}
			for(CEntity *pEnt = pGameServer->m_World.FindFirst(CGameWorld::ENTTYPE_LASER); pEnt; pEnt = pEnt->m_pNextTypeEntity)
			{
				EntityRecord R;
				if(auto *pLaser = dynamic_cast<CLaser *>(pEnt))
				{
					R.Kind = 1;
					R.OwnerClientId = pLaser->m_Owner;
					R.WeaponType = pLaser->m_Type;
					R.PosX = pLaser->m_Pos.x;
					R.PosY = pLaser->m_Pos.y;
					R.DirX = pLaser->m_Dir.x;
					R.DirY = pLaser->m_Dir.y;
					R.StartTick = pLaser->m_EvalTick;
					R.Extra = pLaser->m_Bounces;
				}
				// NOTE on field availability for the 5 classes below: the `#define private
				// public` / `#define protected public` trick (see top of file) is a textual
				// substitution -- it only rewrites a literal `private:`/`protected:` LABEL that
				// appears in the header. A class whose members are declared before ANY explicit
				// access-label sits in C++'s *implicit* default-private region, which has no such
				// literal token to rewrite, so the macro does nothing for it. `CDoor`, `CDragger`,
				// `CDraggerBeam` and `CGun` (checked against this exact build's headers) declare
				// ALL their interesting state (m_Direction/m_Length, m_Core/m_Strength/
				// m_IgnoreWalls/m_EvalTick, m_ForClientId, m_Freeze/m_Explosive) this way -- so
				// none of it is reachable without editing DDNet's own headers, which we
				// deliberately never do (would stop this from running the real, unmodified game
				// code -- the entire point of Oracle B). `CLight` is the one exception: its
				// motion-state fields (m_Rotation/m_To/m_Core/m_Tick) are implicit-private and
				// unreachable the same way, but its *configuration* fields (m_CurveLength,
				// m_LengthL, m_AngularSpeed, m_Speed, m_Length) sit after an explicit `public:`
				// label and so are genuinely public -- dumped below.
				// All 5 classes inherit `CEntity::m_Pos` (declared public in entity.h itself, no
				// macro needed), so position is always recorded; direction/extra are 0 where the
				// backing field is unreachable, documented per-kind in docs/formats.md section 8.2.
				else if(auto *pDoor = dynamic_cast<CDoor *>(pEnt))
				{
					R.Kind = 2;
					R.OwnerClientId = -1;
					R.WeaponType = 0;
					R.PosX = pDoor->m_Pos.x;
					R.PosY = pDoor->m_Pos.y;
					R.DirX = 0; // m_Direction is implicit-private, unreachable -- see note above.
					R.DirY = 0;
					R.StartTick = 0;
					R.Extra = 0; // m_Length is implicit-private, unreachable -- see note above.
				}
				else if(auto *pDragger = dynamic_cast<CDragger *>(pEnt))
				{
					R.Kind = 3;
					R.OwnerClientId = -1;
					R.WeaponType = 0; // m_IgnoreWalls is implicit-private, unreachable.
					R.PosX = pDragger->m_Pos.x;
					R.PosY = pDragger->m_Pos.y;
					R.DirX = 0; // m_Core is implicit-private, unreachable.
					R.DirY = 0;
					R.StartTick = 0; // m_EvalTick is implicit-private, unreachable.
					R.Extra = 0; // m_Strength is implicit-private, unreachable.
				}
				else if(auto *pBeam = dynamic_cast<CDraggerBeam *>(pEnt))
				{
					R.Kind = 4;
					R.OwnerClientId = -1; // m_ForClientId is implicit-private, unreachable; GetOwnerId() is not overridden either (returns -1).
					R.WeaponType = 0;
					R.PosX = pBeam->m_Pos.x;
					R.PosY = pBeam->m_Pos.y;
					R.DirX = 0;
					R.DirY = 0;
					R.StartTick = 0; // m_EvalTick is implicit-private, unreachable.
					R.Extra = 0; // m_Strength is implicit-private, unreachable.
				}
				else if(auto *pGun = dynamic_cast<CGun *>(pEnt))
				{
					R.Kind = 5;
					R.OwnerClientId = -1;
					R.WeaponType = 0; // m_Freeze/m_Explosive are implicit-private, unreachable.
					R.PosX = pGun->m_Pos.x;
					R.PosY = pGun->m_Pos.y;
					R.DirX = 0; // m_Core is implicit-private, unreachable.
					R.DirY = 0;
					R.StartTick = 0; // m_EvalTick is implicit-private, unreachable.
					R.Extra = 0;
				}
				else if(auto *pLight = dynamic_cast<CLight *>(pEnt))
				{
					R.Kind = 6;
					R.OwnerClientId = -1;
					R.WeaponType = pLight->m_Length; // genuinely public (see note above); reused here, not a weapon id.
					R.PosX = pLight->m_Pos.x;
					R.PosY = pLight->m_Pos.y;
					R.DirX = pLight->m_AngularSpeed; // genuinely public.
					R.DirY = 0;
					R.StartTick = 0; // m_Core/m_Rotation (actual sweep state) are implicit-private, unreachable.
					R.Extra = pLight->m_Speed; // genuinely public.
				}
				else
				{
					continue; // Should be unreachable -- every class on this slot is handled above.
				}
				Glob.Entities.push_back(R);
			}
		}

		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		{
			int Id = (int)CharIds[Slot];
			CPlayer *pPlayer = pGameServer->m_apPlayers[Id];
			CCharacter *pChar = pPlayer ? pPlayer->m_pCharacter : nullptr;
			bool Alive = pChar && pChar->m_Alive;

			CoreStateFields &Core = AllCore[Tick][Slot];
			DDRaceStateFields &DD = AllDDRace[Tick][Slot];

			if(Alive)
			{
				const CCharacterCore &Cr = pChar->m_Core;
				Core.PosX = Cr.m_Pos.x;
				Core.PosY = Cr.m_Pos.y;
				Core.VelX = Cr.m_Vel.x;
				Core.VelY = Cr.m_Vel.y;
				Core.HookPosX = Cr.m_HookPos.x;
				Core.HookPosY = Cr.m_HookPos.y;
				Core.HookDirX = Cr.m_HookDir.x;
				Core.HookDirY = Cr.m_HookDir.y;
				Core.HookTeleBaseX = Cr.m_HookTeleBase.x;
				Core.HookTeleBaseY = Cr.m_HookTeleBase.y;
				Core.HookTick = Cr.m_HookTick;
				Core.HookState = Cr.m_HookState;
				Core.HookedPlayer = Cr.HookedPlayer();
				Core.ActiveWeapon = Cr.m_ActiveWeapon;
				Core.NewHook = Cr.m_NewHook ? 1 : 0;
				Core.Jumped = Cr.m_Jumped;
				Core.JumpedTotal = Cr.m_JumpedTotal;
				Core.Jumps = Cr.m_Jumps;
				Core.Direction = Cr.m_Direction;
				Core.Angle = Cr.m_Angle;
				Core.TriggeredEvents = Cr.m_TriggeredEvents;
				Core.Colliding = Cr.m_Colliding;
				Core.LeftWall = Cr.m_LeftWall ? 1 : 0;
				Core.MoveRestrictions = Cr.m_MoveRestrictions;
				Core.Solo = Cr.m_Solo ? 1 : 0;
				Core.CollisionDisabled = Cr.m_CollisionDisabled ? 1 : 0;
				Core.EndlessHook = Cr.m_EndlessHook ? 1 : 0;
				Core.HookHitDisabled = Cr.m_HookHitDisabled ? 1 : 0;

				DD.Alive = 1;
				DD.FreezeTime = pChar->m_FreezeTime;
				DD.IsInFreeze = Cr.m_IsInFreeze ? 1 : 0;
				DD.DeepFrozen = Cr.m_DeepFrozen ? 1 : 0;
				DD.LiveFrozen = Cr.m_LiveFrozen ? 1 : 0;
				DD.FrozenLastTick = pChar->m_FrozenLastTick ? 1 : 0;
				DD.ReloadTimer = pChar->m_ReloadTimer;
				DD.AttackTick = pChar->m_AttackTick;
				DD.QueuedWeapon = pChar->m_QueuedWeapon;
				DD.LastWeapon = pChar->m_LastWeapon;
				DD.WeaponGotMask = 0;
				for(int w = 0; w < NUM_WEAPONS; w++)
				{
					if(Cr.m_aWeapons[w].m_Got)
						DD.WeaponGotMask |= (1 << w);
					DD.WeaponAmmo[w] = Cr.m_aWeapons[w].m_Ammo;
					DD.WeaponAmmoRegenStart[w] = Cr.m_aWeapons[w].m_AmmoRegenStart;
				}
				DD.NinjaActivationTick = Cr.m_Ninja.m_ActivationTick;
				DD.NinjaCurrentMoveTime = Cr.m_Ninja.m_CurrentMoveTime;
				DD.NinjaOldVelAmount = Cr.m_Ninja.m_OldVelAmount;
				DD.NinjaActivationDirX = Cr.m_Ninja.m_ActivationDir.x;
				DD.NinjaActivationDirY = Cr.m_Ninja.m_ActivationDir.y;
				DD.TeleCheckpoint = pChar->m_TeleCheckpoint;
				DD.EndlessJump = Cr.m_EndlessJump ? 1 : 0;
				DD.Jetpack = Cr.m_Jetpack ? 1 : 0;
				DD.Super = Cr.m_Super ? 1 : 0;
				DD.Invincible = Cr.m_Invincible ? 1 : 0;
				DD.HammerHitDisabled = Cr.m_HammerHitDisabled ? 1 : 0;
				DD.GrenadeHitDisabled = Cr.m_GrenadeHitDisabled ? 1 : 0;
				DD.LaserHitDisabled = Cr.m_LaserHitDisabled ? 1 : 0;
				DD.ShotgunHitDisabled = Cr.m_ShotgunHitDisabled ? 1 : 0;
				DD.HasTelegunGun = Cr.m_HasTelegunGun ? 1 : 0;
				DD.HasTelegunGrenade = Cr.m_HasTelegunGrenade ? 1 : 0;
				DD.HasTelegunLaser = Cr.m_HasTelegunLaser ? 1 : 0;
				DD.Team = pChar->Team();
				DD.StrongWeakId = pChar->m_StrongWeakId;
				DD.FreezeStart = Cr.m_FreezeStart;
				DD.FreezeEnd = Cr.m_FreezeEnd;
				DD.TuneZone = pChar->m_TuneZone;
				DD.NumInputs = pChar->m_NumInputs;
				DD.LastRefillJumps = pChar->m_LastRefillJumps ? 1 : 0;
				DD.DDRaceState = (int32_t)pChar->m_DDRaceState;
				DD.StartTime = pChar->m_StartTime;
				DD.DieTick = pPlayer->m_DieTick;
				DD.Spawning = pPlayer->m_Spawning ? 1 : 0;
				DD.PreviousDieTick = pPlayer->m_PreviousDieTick;

				// F13 (round-2 review, fixing F3's own first attempt): exact teleport detection
				// -- the character "teleported this tick" iff BOTH (a) its position BEFORE this
				// tick's movement (still held in `PrevPositions[Slot]` at this point) sat on a
				// tile that actually RELOCATES THE CHARACTER (`IsTeleport`/`IsEvilTeleport`/
				// `IsCheckTeleport`/`IsCheckEvilTeleport` -- `IsTeleportHook` deliberately
				// EXCLUDED: `TILE_TELEINHOOK` only ever redirects the HOOK, per
				// `game/gamecore.cpp:399`, never the character standing on it -- counting it
				// inflated `teleport_ticks` for every tick anyone merely stood on such a tile),
				// AND (b) the character's position actually jumped by more than any plausible
				// single-tick movement this tick -- guards against a check-teleport tile that
				// didn't actually fire (e.g. `TELECHECKIN` with no matching checkpoint reached
				// yet) still being counted as if it had. DDRace teleports happen inside the SAME
				// tick the character is on the tile (`HandleTiles`), so the position already
				// recorded after `OnTick()` is the destination -- checking the PREVIOUS tick's
				// tile (not this tick's) is what actually observes "was on a teleporter".
				{
					CCollision *pCol = pGameServer->Collision();
					int32_t PrevIdx = pCol->GetPureMapIndex((float)PrevPositions[Slot].first, (float)PrevPositions[Slot].second);
					bool OnTeleportTile = pCol->IsTeleport(PrevIdx) || pCol->IsEvilTeleport(PrevIdx) ||
						pCol->IsCheckTeleport(PrevIdx) || pCol->IsCheckEvilTeleport(PrevIdx);
					if(OnTeleportTile)
					{
						float Ddx = Cr.m_Pos.x - (float)PrevPositions[Slot].first;
						float Ddy = Cr.m_Pos.y - (float)PrevPositions[Slot].second;
						if(Ddx * Ddx + Ddy * Ddy > (200.0f * 200.0f))
							Cov.TeleportTicks++;
					}
				}
				PrevPositions[Slot] = {(int32_t)Cr.m_Pos.x, (int32_t)Cr.m_Pos.y};

				// Coverage counters (F3: hammer swing/hit fixed, teleport now exact above;
				// F6/F7: several new counters derived from the newly-added trace fields).
				Cov.CharacterTicks++;
				if(DD.IsInFreeze)
					Cov.FrozenTicks++;
				if(DD.DeepFrozen)
					Cov.DeepFrozenTicks++;
				if(DD.LiveFrozen)
					Cov.LiveFrozenTicks++;
				// `GetMapIndex` returns -1 for a perfectly ordinary open-air cell (any tile
				// `TileExists` doesn't consider "interesting"), which `IsSpeedup` asserts
				// against -- `GetPureMapIndex` is the always-clamped-non-negative variant meant
				// for exactly this kind of "what's at this position" query.
				if(pGameServer->Collision()->IsSpeedup(pGameServer->Collision()->GetPureMapIndex(Core.PosX, Core.PosY)))
					Cov.SpeedupTicks++;
				if(Core.MoveRestrictions != 0)
					Cov.StopperTicks++;
				if(DD.TuneZone != 0)
					Cov.TuneZoneTicks++;
				if(Core.Solo)
					Cov.SoloTicks++;
				if(DD.Jetpack)
					Cov.JetpackTicks++;
				if(DD.EndlessJump)
					Cov.EndlessJumpTicks++;
				if(DD.Super)
					Cov.SuperTicks++;
				if(Core.CollisionDisabled)
					Cov.CollisionDisabledTicks++;
				if(Core.HookHitDisabled)
					Cov.HookHitDisabledTicks++;
				if(DD.Team != 0)
					Cov.NonZeroTeamTicks++;
				// F3: hammer swing/hit -- `m_AttackTick` is stamped during
				// `OnClientPredictedEarlyInput` (before `m_CurrentGameTick++`), so a NEW swing
				// this tick shows up as `AttackTick` CHANGING to a value that is one behind the
				// just-dumped `GameTick` -- comparing against the previous *recorded* AttackTick
				// (not against `GameTick` itself, which was the old, wrong check) catches every
				// swing exactly once. Hit-vs-miss is read off `reload_timer`, replicating
				// `FireWeapon()`'s own arithmetic (character.cpp) against the SAME tuning this
				// character actually fired under (`TuningList()[TuneZone]`), so tune overrides
				// don't desync the classification. `HandleWeapons()` (character.cpp:668-670,
				// `if(m_ReloadTimer) m_ReloadTimer--;`) unconditionally decrements it once EVERY
				// tick, including the very tick `FireWeapon()` (called earlier the same tick,
				// from the early-input phase) just set it -- so the value observed here, after
				// `OnTick()` has fully run, is always the raw formula's result MINUS ONE. An
				// earlier version of this check compared against the raw (un-decremented)
				// value and so classified every real hit as a miss (verified by inspection: the
				// initial round-2 corpus's few detected "misses" all showed
				// `reload_timer == raw_hit_delay - 1`, i.e. were actually hits).
				if(DD.AttackTick != PrevAttackTick[Slot])
				{
					PrevAttackTick[Slot] = DD.AttackTick;
					if(Core.ActiveWeapon == WEAPON_HAMMER)
					{
						Cov.HammerSwings++;
						const CTuningParams &Tn = pGameServer->TuningList()[DD.TuneZone];
						int32_t HitTicks = (int32_t)((float)Tn.m_HammerHitFireDelay * (float)SERVER_TICK_SPEED / 1000.0f) - 1;
						if(DD.ReloadTimer == HitTicks)
							Cov.HammerHits++;
					}
				}
				// COREEVENT_HOOK_ATTACH_PLAYER (0x08, gamecore.h) is set exactly on the tick a
				// hook attaches to another character -- an exact edge, no state tracking needed.
				if(Cr.m_TriggeredEvents & COREEVENT_HOOK_ATTACH_PLAYER)
					Cov.HookGrabs++;
			}
			else
			{
				// F8 fix: formats.md documents a dead character's row as a COPY of the previous
				// tick's fields (position/state frozen at "where it died"), not zeroed -- the
				// code used to zero `DD` (and only copy `Core`), disagreeing with the docs. Copy
				// both, then re-clear the three fields that must reflect THIS tick regardless.
				if(Tick > 0)
				{
					DD = AllDDRace[Tick - 1][Slot];
					Core = AllCore[Tick - 1][Slot];
				}
				else
				{
					DD = DDRaceStateFields{};
					Core = CoreStateFields{};
				}
				DD.Alive = 0;
				DD.DiedThisTick = 0;
				DD.RespawnedThisTick = 0;
			}

			bool NowInFreeze = DD.IsInFreeze != 0;
			if(NowInFreeze && !WasInFreeze[Slot])
				Cov.FreezeEntries++;
			if(!NowInFreeze && WasInFreeze[Slot])
				Cov.FreezeExits++;
			WasInFreeze[Slot] = NowInFreeze;

			// F12 fallout (found while verifying this round's fix, not itself an F-numbered
			// finding): the real `OnKillNetMessage` path -- and, in general, any death whose
			// `sv_kill_delay`/protection checks pass -- calls `Respawn()` in the SAME tick as the
			// kill (per the F12 orchestrator decision: "the harness applies it exactly like the
			// real server", and the real server's own `Respawn()`/`TryRespawn()` doesn't wait for
			// a tick boundary). A death immediately followed by a same-tick respawn leaves
			// `Alive` (sampled once, after `OnTick()` has fully run) TRUE both before and after
			// this tick -- invisible to a pure before/after `Alive`-flag comparison, which would
			// silently under-report `died`/`respawned` coverage for every kill that has a valid
			// spawn point (confirmed: a kill log line with `died=0` in the same run's coverage
			// summary before this fix). `CPlayer::m_DieTick`/a fresh `CCharacter::m_SpawnTick`
			// are set unconditionally on death/(re)spawn respectively, REGARDLESS of whether the
			// old character object got destroyed and a new one created within the same tick, so
			// comparing them against this tick's own number catches the same-tick case too, on
			// top of the ordinary multi-tick-gap case the `Alive`-flag comparison already caught.
			// Tick 0's own initial spawn (from `ForceSpawn`, called before this loop starts) is
			// NOT a "death"/"respawn" -- there is no prior tick to have died in -- so the new
			// SpawnTick-equals-this-tick check (and seeding `PrevDieTick`) are gated on
			// `Tick > 0`, matching this file's existing Tick-0-is-special handling a few lines
			// above (F8's dead-row-copy).
			bool DiedThisTickReal = (WasAlive[Slot] && !Alive) || (Tick > 0 && Alive && DD.DieTick != PrevDieTick[Slot]);
			bool RespawnedThisTickReal = (!WasAlive[Slot] && Alive) || (Tick > 0 && Alive && pChar->m_SpawnTick == Glob.GameTick);
			if(DiedThisTickReal)
			{
				DD.DiedThisTick = 1;
				Cov.Died++;
			}
			if(RespawnedThisTickReal)
			{
				DD.RespawnedThisTick = 1;
				Cov.Respawned++;
			}
			if(Alive)
				PrevDieTick[Slot] = DD.DieTick;
			WasAlive[Slot] = Alive;
		}
	}
	auto End = std::chrono::steady_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();
	double TeeTicks = (double)Ticks * (double)NumChars;
	double Throughput = Seconds > 0 ? TeeTicks / Seconds : 0.0;

	// -------------------------------------------------------------------------------------
	// F5 (round-1 review): `--emit-scenario-v3` -- writes a self-contained (rawmap + scenario
	// v3) pair next to the trace that replays it byte-identically without the original `.map`
	// file or (for real-map mode) this harness's own generator: map2raw extracts the tile
	// layers `CCollision` already holds from `LoadMap()`, and the scenario records the exact
	// per-character team, the exact resolved input this run actually applied every tick
	// (`AllInputs`, already collected above), every `--cfg` file's lines verbatim, and the
	// seed. See docs/formats.md section 9.6 and WriteScenarioV3/ExtractRawMap above.
	// -------------------------------------------------------------------------------------
	if(!EmitScenarioV3Path.empty())
	{
		std::string RawmapOutPath = EmitScenarioV3Path;
		const std::string ScnSuffix = ".scn";
		if(RawmapOutPath.size() >= ScnSuffix.size() && RawmapOutPath.compare(RawmapOutPath.size() - ScnSuffix.size(), ScnSuffix.size(), ScnSuffix) == 0)
			RawmapOutPath.resize(RawmapOutPath.size() - ScnSuffix.size());
		RawmapOutPath += ".rawmap";

		RawMap Extracted = ExtractRawMap(pGameServer->Collision(), pGameServer->Map());
		WriteRawMapFile(Extracted, RawmapOutPath);
		std::array<unsigned char, 32> RawmapSha256 = oracle_sha256::Digest(ReadFile(RawmapOutPath));

		std::vector<CharacterSpawnRaw> OutCharacters(NumChars);
		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		{
			OutCharacters[Slot].Id = CharIds[Slot];
			OutCharacters[Slot].SpawnX = SpawnPositions[Slot].first;
			OutCharacters[Slot].SpawnY = SpawnPositions[Slot].second;
			OutCharacters[Slot].Team = EffectiveTeam[Slot]; // computed once, right after character creation.
		}

		std::vector<std::string> EmbeddedCfgLines;
		for(const std::string &CfgFile : CfgFiles)
		{
			std::vector<unsigned char> Bytes = ReadFile(CfgFile);
			std::string Text((const char *)Bytes.data(), Bytes.size());
			size_t Pos = 0;
			while(Pos <= Text.size())
			{
				size_t Nl = Text.find('\n', Pos);
				std::string Line = Text.substr(Pos, Nl == std::string::npos ? std::string::npos : Nl - Pos);
				while(!Line.empty() && (Line.back() == '\r' || Line.back() == ' ' || Line.back() == '\t'))
					Line.pop_back();
				if(!Line.empty())
					EmbeddedCfgLines.push_back(Line);
				if(Nl == std::string::npos)
					break;
				Pos = Nl + 1;
			}
		}
		if(RawmapMode)
			for(const std::string &Line : S.CfgLines)
				EmbeddedCfgLines.push_back(Line);
		if(RawmapMode && S.NoWeakHook)
			EmbeddedCfgLines.push_back("sv_no_weak_hook 1");

		std::string EmbeddedGeneratorId = GeneratorName.empty() ? std::string(RawmapMode ? "rawmap-scenario-replay/1" : "oracle-server-realmap-gen/1") : GeneratorName;
		WriteScenarioV3(EmitScenarioV3Path, RawmapOutPath, RawmapSha256, OutCharacters, AllInputs, EmbeddedCfgLines, EmbeddedGeneratorId, Seed);
		fprintf(stderr, "oracle_server: wrote replayable scenario %s + %s\n", EmitScenarioV3Path.c_str(), RawmapOutPath.c_str());
	}

	// -------------------------------------------------------------------------------------
	// trace-b v1 output (docs/formats.md section 8)
	// -------------------------------------------------------------------------------------
	std::ostringstream Json;
	Json << "{";
	Json << "\"producer\":{\"name\":\"ddnet-oracle-b\",\"version\":\"1\"},";
	Json << "\"ddnet\":{\"tag\":\"20.1\",\"commit\":\"c9d208138f85755521f16a0096b6fe036c5c8698\"},";
	Json << "\"map_sha256\":\"" << oracle_sha256::ToHex(ActualMapSha256) << "\",";
	Json << "\"seed\":" << Seed << ",";
	if(!GeneratorName.empty())
		Json << "\"generator\":\"" << JsonEscape(GeneratorName) << "\",";
	if(RawmapMode)
	{
		Json << "\"mode\":\"rawmap-scenario\",";
		Json << "\"scenario_sha256\":\"" << oracle_sha256::ToHex(ScenarioSha256) << "\"";
	}
	else
	{
		Json << "\"mode\":\"real-map\",";
		Json << "\"real_map_path\":\"" << JsonEscape(RealMapPath) << "\"";
	}
	Json << "}";

	ByteWriter W;
	W.Magic("TRB1");
	W.U32(2); // F14 (round-2 review) bumped this from 1: header gained switch_highest_number/
	// switch_team_count/switch_team_ids, and each row's switch section is no longer
	// self-describing (count moved to the header) -- see docs/formats.md section 8.2.
	W.String32(Json.str());
	W.U32(NumChars);
	for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		W.U32(CharIds[Slot]);
	W.U32((uint32_t)HighestSwitchNumber);
	W.U32((uint32_t)TeamsDumped.size());
	for(int32_t T : TeamsDumped)
		W.I32(T);
	W.U32(Ticks);
	for(uint32_t Tick = 0; Tick < Ticks; Tick++)
	{
		AllGlobal[Tick].Write(W);
		for(uint32_t Slot = 0; Slot < NumChars; Slot++)
		{
			const ResolvedInput &In = AllInputs[Tick][Slot];
			W.I32(In.Direction);
			W.I32(In.TargetX);
			W.I32(In.TargetY);
			W.I32(In.Jump);
			W.I32(In.Fire);
			W.I32(In.Hook);
			W.I32(In.PlayerFlags);
			W.I32(In.WantedWeapon);
			W.I32(In.NextWeapon);
			W.I32(In.PrevWeapon);
			W.I32(In.Kill); // F12 (round-2 review): recorded per tick, not just applied silently.
			AllCore[Tick][Slot].Write(W);
			AllDDRace[Tick][Slot].Write(W);
		}
	}
	WriteFile(OutPath, W.Data());

	fprintf(stderr, "oracle_server: %u ticks, %u characters, %.0f tee-ticks/s, wrote %s (%zu bytes)\n",
		Ticks, NumChars, Throughput, OutPath.c_str(), W.Data().size());
	fprintf(stderr,
		"oracle_server: coverage character_ticks=%llu frozen=%llu deep_frozen=%llu live_frozen=%llu "
		"freeze_entries=%llu freeze_exits=%llu speedup=%llu stopper=%llu tune_zone=%llu teleport=%llu "
		"hammer_swings=%llu hammer_hits=%llu hook_grabs=%llu switch_toggles=%llu solo=%llu jetpack=%llu "
		"endless_jump=%llu super=%llu collision_disabled=%llu hook_hit_disabled=%llu nonzero_team=%llu "
		"died=%llu respawned=%llu\n",
		(unsigned long long)Cov.CharacterTicks, (unsigned long long)Cov.FrozenTicks, (unsigned long long)Cov.DeepFrozenTicks,
		(unsigned long long)Cov.LiveFrozenTicks, (unsigned long long)Cov.FreezeEntries, (unsigned long long)Cov.FreezeExits,
		(unsigned long long)Cov.SpeedupTicks, (unsigned long long)Cov.StopperTicks, (unsigned long long)Cov.TuneZoneTicks,
		(unsigned long long)Cov.TeleportTicks, (unsigned long long)Cov.HammerSwings, (unsigned long long)Cov.HammerHits,
		(unsigned long long)Cov.HookGrabs, (unsigned long long)Cov.SwitchToggles, (unsigned long long)Cov.SoloTicks,
		(unsigned long long)Cov.JetpackTicks, (unsigned long long)Cov.EndlessJumpTicks, (unsigned long long)Cov.SuperTicks,
		(unsigned long long)Cov.CollisionDisabledTicks, (unsigned long long)Cov.HookHitDisabledTicks,
		(unsigned long long)Cov.NonZeroTeamTicks, (unsigned long long)Cov.Died, (unsigned long long)Cov.Respawned);

	if(!CoverageOutPath.empty())
	{
		std::ostringstream Cj;
		Cj << "{\"character_ticks\":" << Cov.CharacterTicks << ",\"frozen_ticks\":" << Cov.FrozenTicks
		   << ",\"deep_frozen_ticks\":" << Cov.DeepFrozenTicks << ",\"live_frozen_ticks\":" << Cov.LiveFrozenTicks
		   << ",\"freeze_entries\":" << Cov.FreezeEntries << ",\"freeze_exits\":" << Cov.FreezeExits
		   << ",\"speedup_ticks\":" << Cov.SpeedupTicks << ",\"stopper_ticks\":" << Cov.StopperTicks
		   << ",\"tune_zone_ticks\":" << Cov.TuneZoneTicks << ",\"teleport_ticks\":" << Cov.TeleportTicks
		   << ",\"hammer_swings\":" << Cov.HammerSwings << ",\"hammer_hits\":" << Cov.HammerHits
		   << ",\"hook_grabs\":" << Cov.HookGrabs << ",\"switch_toggles\":" << Cov.SwitchToggles
		   << ",\"solo_ticks\":" << Cov.SoloTicks << ",\"jetpack_ticks\":" << Cov.JetpackTicks
		   << ",\"endless_jump_ticks\":" << Cov.EndlessJumpTicks << ",\"super_ticks\":" << Cov.SuperTicks
		   << ",\"collision_disabled_ticks\":" << Cov.CollisionDisabledTicks
		   << ",\"hook_hit_disabled_ticks\":" << Cov.HookHitDisabledTicks
		   << ",\"nonzero_team_ticks\":" << Cov.NonZeroTeamTicks
		   << ",\"died\":" << Cov.Died << ",\"respawned\":" << Cov.Respawned << "}";
		std::string S2 = Cj.str();
		WriteFile(CoverageOutPath, std::vector<unsigned char>(S2.begin(), S2.end()));
	}

	// -------------------------------------------------------------------------------------
	// Consistency-with-Oracle-A check (acceptance criterion 7): reads an Oracle A trace v1
	// file (docs/formats.md section 6) directly and compares its 28 core fields, tick by tick
	// and character by character (matched by character_ids order, which both oracles derive
	// the same way from the same scenario file -- docs/formats.md section 5.1), against the
	// core fields this run just computed (AllCore above, which reuses the exact same field
	// order as Oracle A's `CharacterCoreState`). Bitwise float compare (`memcmp`), matching
	// `ddai_trace::trace::diff`'s policy (docs/formats.md section 6.4).
	// -------------------------------------------------------------------------------------
	if(!CompareOracleAPath.empty())
	{
		std::vector<unsigned char> AB = ReadFile(CompareOracleAPath);
		ByteReader AR(AB.data(), AB.size());
		AR.ExpectMagic("TRC1");
		uint32_t AVersion = AR.U32("version");
		if(AVersion != 1)
			Fail("--compare-oracle-a: unsupported trace v1 version " + std::to_string(AVersion));
		uint32_t MetaLen = AR.U32("metadata_len");
		AR.Take(MetaLen, "metadata_json"); // unused here -- scenario_sha256 equality is the caller's job.
		uint32_t ACharCount = AR.U32("character_count");
		std::vector<uint32_t> AIds(ACharCount);
		for(uint32_t i = 0; i < ACharCount; i++)
			AIds[i] = AR.U32("character id");
		uint32_t ATicks = AR.U32("tick_count");

		if(ACharCount != NumChars || ATicks != Ticks)
			Fail("--compare-oracle-a: shape mismatch (characters " + std::to_string(ACharCount) + " vs " + std::to_string(NumChars) +
				", ticks " + std::to_string(ATicks) + " vs " + std::to_string(Ticks) + ")");
		for(uint32_t i = 0; i < ACharCount; i++)
			if(AIds[i] != CharIds[i])
				Fail("--compare-oracle-a: character_ids order mismatch at slot " + std::to_string(i));

		uint64_t Mismatches = 0;
		bool ReportedFirst = false;
		for(uint32_t Tick = 0; Tick < ATicks; Tick++)
		{
			for(uint32_t Slot = 0; Slot < ACharCount; Slot++)
			{
				// Applied input (10 i32) -- skip, we only compare state below.
				for(int k = 0; k < 10; k++)
					AR.I32("input field");
				CoreStateFields A{};
				A.PosX = AR.F32("pos_x");
				A.PosY = AR.F32("pos_y");
				A.VelX = AR.F32("vel_x");
				A.VelY = AR.F32("vel_y");
				A.HookPosX = AR.F32("hook_pos_x");
				A.HookPosY = AR.F32("hook_pos_y");
				A.HookDirX = AR.F32("hook_dir_x");
				A.HookDirY = AR.F32("hook_dir_y");
				A.HookTeleBaseX = AR.F32("hook_tele_base_x");
				A.HookTeleBaseY = AR.F32("hook_tele_base_y");
				A.HookTick = AR.I32("hook_tick");
				A.HookState = AR.I32("hook_state");
				A.HookedPlayer = AR.I32("hooked_player");
				A.ActiveWeapon = AR.I32("active_weapon");
				A.NewHook = AR.I32("new_hook");
				A.Jumped = AR.I32("jumped");
				A.JumpedTotal = AR.I32("jumped_total");
				A.Jumps = AR.I32("jumps");
				A.Direction = AR.I32("direction");
				A.Angle = AR.I32("angle");
				A.TriggeredEvents = AR.I32("triggered_events");
				A.Colliding = AR.I32("colliding");
				A.LeftWall = AR.I32("left_wall");
				A.MoveRestrictions = AR.I32("move_restrictions");
				A.Solo = AR.I32("solo");
				A.CollisionDisabled = AR.I32("collision_disabled");
				A.EndlessHook = AR.I32("endless_hook");
				A.HookHitDisabled = AR.I32("hook_hit_disabled");

				// `active_weapon` is EXCLUDED from this comparison on purpose (not a bug): Oracle
				// A's core-only tick never sets it (docs/formats.md section 5.6 -- it stays 0,
				// `Reset()`'s zero-init, for the entire run), while Oracle B goes through the
				// real `CCharacter::Spawn()` (character.cpp:95, `m_Core.m_ActiveWeapon =
				// WEAPON_GUN`) and `HandleWeaponSwitch()`/`DoWeaponSwitch()` afterwards -- this
				// field is guaranteed to differ from Oracle A on essentially every tick of every
				// scenario, by construction, independent of any actual physics divergence. Every
				// other of the 28 core fields IS compared, including velocity (so a hammer/
				// weapon knockback impulse -- a genuine DDRace-level difference outside what
				// "arena, fire never pressed" scenarios are supposed to exercise -- still shows
				// up here if the input generator that produced the compared scenario didn't
				// actually keep fire off the whole run).
				const CoreStateFields &B = AllCore[Tick][Slot];
				const char *pFirstDiffName = nullptr;
				float FirstDiffA = 0, FirstDiffB = 0;
				auto CheckF = [&](const char *Name, float Av, float Bv) {
					if(memcmp(&Av, &Bv, sizeof(float)) != 0 && !pFirstDiffName)
					{
						pFirstDiffName = Name;
						FirstDiffA = Av;
						FirstDiffB = Bv;
					}
					return memcmp(&Av, &Bv, sizeof(float)) == 0;
				};
				auto CheckI = [&](const char *Name, int32_t Av, int32_t Bv) {
					if(Av != Bv && !pFirstDiffName)
					{
						pFirstDiffName = Name;
						FirstDiffA = (float)Av;
						FirstDiffB = (float)Bv;
					}
					return Av == Bv;
				};
				bool Same =
					CheckF("pos_x", A.PosX, B.PosX) & CheckF("pos_y", A.PosY, B.PosY) &
					CheckF("vel_x", A.VelX, B.VelX) & CheckF("vel_y", A.VelY, B.VelY) &
					CheckF("hook_pos_x", A.HookPosX, B.HookPosX) & CheckF("hook_pos_y", A.HookPosY, B.HookPosY) &
					CheckF("hook_dir_x", A.HookDirX, B.HookDirX) & CheckF("hook_dir_y", A.HookDirY, B.HookDirY) &
					CheckF("hook_tele_base_x", A.HookTeleBaseX, B.HookTeleBaseX) &
					CheckF("hook_tele_base_y", A.HookTeleBaseY, B.HookTeleBaseY) &
					CheckI("hook_tick", A.HookTick, B.HookTick) & CheckI("hook_state", A.HookState, B.HookState) &
					CheckI("hooked_player", A.HookedPlayer, B.HookedPlayer) &
					CheckI("new_hook", A.NewHook, B.NewHook) & CheckI("jumped", A.Jumped, B.Jumped) &
					CheckI("jumped_total", A.JumpedTotal, B.JumpedTotal) & CheckI("jumps", A.Jumps, B.Jumps) &
					CheckI("direction", A.Direction, B.Direction) & CheckI("angle", A.Angle, B.Angle) &
					CheckI("triggered_events", A.TriggeredEvents, B.TriggeredEvents) &
					CheckI("colliding", A.Colliding, B.Colliding) & CheckI("left_wall", A.LeftWall, B.LeftWall) &
					CheckI("move_restrictions", A.MoveRestrictions, B.MoveRestrictions) &
					CheckI("solo", A.Solo, B.Solo) & CheckI("collision_disabled", A.CollisionDisabled, B.CollisionDisabled) &
					CheckI("endless_hook", A.EndlessHook, B.EndlessHook) &
					CheckI("hook_hit_disabled", A.HookHitDisabled, B.HookHitDisabled);
				if(!Same)
				{
					Mismatches++;
					if(!ReportedFirst)
					{
						ReportedFirst = true;
						fprintf(stderr, "oracle_server: --compare-oracle-a: first mismatch at tick=%u id=%u: field=%s A=%f B=%f\n",
							Tick, CharIds[Slot], pFirstDiffName ? pFirstDiffName : "?", FirstDiffA, FirstDiffB);
					}
				}
			}
		}
		fprintf(stderr, "oracle_server: --compare-oracle-a: %llu / %llu (tick,character) core-field mismatches (active_weapon excluded -- see comment)\n",
			(unsigned long long)Mismatches, (unsigned long long)((uint64_t)ATicks * ACharCount));
		if(Mismatches > 0)
		{
			g_Config = ConfigBackup;
			return 2;
		}
	}

	g_Config = ConfigBackup;
	return 0;
}
