// Oracle A: reads a rawmap v1 + scenario v1 file (own small readers, per docs/formats.md), runs
// the scenario through DDNet 20.1's REAL CCharacterCore/CCollision/CLayers (compiled from the
// sources fetch.sh fetches into build/, never committed to this repository — this file is
// DDNet-AI's own original code, GPL-3.0-only like the rest of this repo, and does not copy any
// DDNet source), core-level only: no CCharacter/DDRace logic (no freeze, no tile effects, no
// weapons — that is Oracle B, a later task). Writes a trace v1 file with the resulting
// CCharacterCore state after every tick, for every character.
//
// Usage: oracle_core <rawmap-file> <scenario-file> <trace-out-file> [--generator NAME --seed N]
// `--generator`/`--seed` are optional metadata annotations only (see docs/formats.md's
// `scenario` metadata field) — they do not change what is simulated; the scenario's own bytes
// (and their sha256, always included) are what is actually replayed.
//
// See tools/ddnet-oracle/README.md (in Russian) for build/run instructions and
// docs/formats.md for the exact rawmap/scenario/trace byte layouts this file reads and writes.

#include "sha256.h"

#include <engine/map.h>
#include <engine/shared/config.h>
#include <game/collision.h>
#include <game/gamecore.h>
#include <game/layers.h>
#include <game/mapitems.h>
#include <game/teamscore.h>

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <strings.h>
#include <vector>

// -------------------------------------------------------------------------------------------
// Stub externs: gamecore.cpp/collision.cpp/layers.cpp/teamscore.cpp reference these but this
// harness never needs their real behavior (no config files, no logging, no UTF-8 validation of
// untrusted strings — every string here comes from our own generator or our own map/scenario
// readers). Matches data/research/physics-scratch/harness_proto.cpp, the phase-0 prototype this
// file supersedes.
// -------------------------------------------------------------------------------------------
CConfig g_Config;

extern "C" void dbg_assert_imp(const char *filename, int line, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "assert %s:%d: ", filename, line);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	abort();
}
int str_comp_nocase(const char *a, const char *b) { return strcasecmp(a, b); }
int str_format(char *b, int n, const char *f, ...)
{
	va_list ap;
	va_start(ap, f);
	int r = vsnprintf(b, n, f, ap);
	va_end(ap);
	return r;
}
int str_length(const char *s) { return (int)strlen(s); }
int str_utf8_check(const char *) { return 1; }

namespace
{

// =============================================================================================
// Tiny binary reader/writer, mirroring crates/ddai-trace/src/io.rs field-for-field.
// =============================================================================================

[[noreturn]] void Fail(const std::string &Msg)
{
	fprintf(stderr, "oracle_core: %s\n", Msg.c_str());
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
	void Bytes(unsigned char *Out, size_t N, const char *Ctx)
	{
		memcpy(Out, Take(N, Ctx), N);
	}
	size_t Remaining() const { return m_Len - m_Pos; }

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

// =============================================================================================
// rawmap v1 (see docs/formats.md and crates/ddai-trace/src/rawmap.rs)
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
	// Settings strings follow; Oracle A doesn't use map settings (see docs/formats.md), so they
	// are intentionally not parsed.
	return M;
}

// =============================================================================================
// scenario v2 (see docs/formats.md and crates/ddai-trace/src/scenario.rs)
// =============================================================================================

struct TuningOverride
{
	std::string Name;
	int32_t ValueX100 = 0;
};

struct CharacterSpawn
{
	uint32_t Id = 0;
	int32_t SpawnX = 0, SpawnY = 0;
};

// The *stored* per-tick input recipe (scenario v2) — `AimSlot >= 0` means `TargetX`/`TargetY`
// are noise added to a live vector, resolved by `ResolveInput` below, not the raw applied
// input. Mirrors `crates/ddai-trace/src/scenario.rs`'s `ScenarioInput`.
struct ScenarioInputRecord
{
	int32_t Direction = 0, TargetX = 0, TargetY = 0, AimSlot = -1, Jump = 0, Fire = 0, Hook = 0;
	int32_t PlayerFlags = 0, WantedWeapon = 0, NextWeapon = 0, PrevWeapon = 0;
};

struct Scenario
{
	bool MapRefIsRecipe = true;
	std::string MapRefString;
	std::array<unsigned char, 32> MapSha256{};
	bool NoWeakHook = false;
	std::vector<TuningOverride> TuningOverrides;
	std::vector<CharacterSpawn> Characters;
	std::vector<std::vector<ScenarioInputRecord>> Inputs; // Inputs[tick][slot]
};

// Rejects a character id >= MAX_CLIENTS or a duplicate id (review round 1, finding F5) and an
// `AimSlot` outside `-1..characters.size()` (finding F3) — mirrors
// `crates/ddai-trace/src/scenario.rs`'s `Scenario::validate`.
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
	if(Version != 2)
		Fail("unsupported scenario version " + std::to_string(Version) + " (expected 2)");
	Scenario S;
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
		CharacterSpawn C;
		C.Id = R.U32("character id");
		C.SpawnX = R.I32("character spawn x");
		C.SpawnY = R.I32("character spawn y");
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
		}
	}
	ValidateScenario(S);
	return S;
}

// The concrete, absolute input actually applied to `CCharacterCore::m_Input` for one tick — the
// *resolved* form (mirrors `crates/ddai-trace/src/scenario.rs`'s `PlayerInput`).
struct ResolvedInput
{
	int32_t Direction = 0, TargetX = 0, TargetY = 0, Jump = 0, Fire = 0, Hook = 0;
	int32_t PlayerFlags = 0, WantedWeapon = 0, NextWeapon = 0, PrevWeapon = 0;
};

// Implements the exact same "aim mode" resolution as
// `crates/ddai-trace/src/scenario.rs`'s `resolve_input` (review round 1, finding F3) — see that
// function's doc comment and docs/formats.md §2.1 for the full algorithm description. All
// arithmetic is `int32_t` (never float): `PrevPositions` entries are always exactly integer
// (a spawn position, or read back from a `CCharacterCore::m_Pos` that `Quantize()` already
// rounded to whole pixels), so there is nothing for float rounding to disagree with Rust about.
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
	return Out;
}

// =============================================================================================
// In-memory IMap + CLayers/CCollision setup, generalized from
// data/research/physics-scratch/harness_proto.cpp's CFakeMap to cover every optional layer.
// =============================================================================================

class CFakeMap : public IMap
{
public:
	std::vector<std::vector<unsigned char>> m_vItems;
	std::vector<int> m_vItemTypes;
	std::vector<std::vector<unsigned char>> m_vData;

	int GetDataSize(int Index) const override { return (int)m_vData[Index].size(); }
	void *GetData(int Index) override { return m_vData[Index].data(); }
	void *GetDataSwapped(int Index) override { return GetData(Index); }
	const char *GetDataString(int) override { return ""; }
	void UnloadData(int) override {}
	int NumData() const override { return (int)m_vData.size(); }
	int GetItemSize(int Index) override { return (int)m_vItems[Index].size(); }
	void *GetItem(int Index, int *pType, int *pId, CUuid *) override
	{
		if(pType)
			*pType = m_vItemTypes[Index];
		if(pId)
			*pId = 0;
		return m_vItems[Index].data();
	}
	void GetType(int Type, int *pStart, int *pNum) override
	{
		*pStart = 0;
		*pNum = 0;
		bool Found = false;
		for(int i = 0; i < (int)m_vItemTypes.size(); i++)
			if(m_vItemTypes[i] == Type)
			{
				if(!Found)
					*pStart = i;
				Found = true;
				(*pNum)++;
			}
	}
	int FindItemIndex(int, int) override { return -1; }
	void *FindItem(int, int) override { return nullptr; }
	int NumItems() const override { return (int)m_vItems.size(); }
	bool Load(const char *, IStorage *, const char *, int) override { return true; }
	bool Load(IStorage *, const char *, int) override { return true; }
	void Unload() override {}
	bool IsLoaded() const override { return true; }
	IOHANDLE File() const override { return nullptr; }
	const char *FullName() const override { return "oracle-a"; }
	const char *BaseName() const override { return "oracle-a"; }
	const char *Path() const override { return "oracle-a"; }
	SHA256_DIGEST Sha256() const override { return SHA256_DIGEST{}; }
	unsigned Crc() const override { return 0; }
	int Size() const override { return 0; }

	template<typename T>
	int PushData(const std::vector<T> &V)
	{
		m_vData.emplace_back((const unsigned char *)V.data(), (const unsigned char *)(V.data() + V.size()));
		return (int)m_vData.size() - 1;
	}
	void PushItem(int Type, const void *P, size_t N)
	{
		m_vItemTypes.push_back(Type);
		m_vItems.emplace_back((const unsigned char *)P, (const unsigned char *)P + N);
	}
};

CMapItemLayerTilemap MakeTilemapItem(int Width, int Height, int Flags)
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
	L.m_aName[0] = L.m_aName[1] = L.m_aName[2] = 0;
	return L;
}

// Builds the in-memory map, CLayers and CCollision for `Map`. `FakeMap`/`Layers`/`Collision`
// must outlive their use — the caller keeps them as local variables that outlive every use
// below, since `CCollision`/`CLayers` only ever store raw pointers into what they're given,
// exactly like the real DDNet server.
void BuildCollision(const RawMap &Map, CFakeMap &FakeMap, CLayers &Layers, CCollision &Collision)
{
	int GameData = FakeMap.PushData(Map.Game);
	int NumLayers = 1;
	int FrontData = -1, TeleData = -1, SpeedupData = -1, SwitchData = -1, TuneData = -1;
	if(Map.HasFront)
	{
		FrontData = FakeMap.PushData(Map.Front);
		NumLayers++;
	}
	if(Map.HasTele)
	{
		TeleData = FakeMap.PushData(Map.Tele);
		NumLayers++;
	}
	if(Map.HasSpeedup)
	{
		SpeedupData = FakeMap.PushData(Map.Speedup);
		NumLayers++;
	}
	if(Map.HasSwitch)
	{
		SwitchData = FakeMap.PushData(Map.Switch);
		NumLayers++;
	}
	if(Map.HasTune)
	{
		TuneData = FakeMap.PushData(Map.Tune);
		NumLayers++;
	}

	CMapItemGroup Group{};
	Group.m_Version = 3;
	Group.m_ParallaxX = Group.m_ParallaxY = 100;
	Group.m_StartLayer = 0;
	Group.m_NumLayers = NumLayers;
	FakeMap.PushItem(MAPITEMTYPE_GROUP, &Group, sizeof(Group));

	CMapItemLayerTilemap GameL = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_GAME);
	GameL.m_Data = GameData;
	FakeMap.PushItem(MAPITEMTYPE_LAYER, &GameL, sizeof(GameL));

	if(Map.HasFront)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_FRONT);
		L.m_Front = FrontData;
		FakeMap.PushItem(MAPITEMTYPE_LAYER, &L, sizeof(L));
	}
	if(Map.HasTele)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_TELE);
		L.m_Tele = TeleData;
		FakeMap.PushItem(MAPITEMTYPE_LAYER, &L, sizeof(L));
	}
	if(Map.HasSpeedup)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_SPEEDUP);
		L.m_Speedup = SpeedupData;
		FakeMap.PushItem(MAPITEMTYPE_LAYER, &L, sizeof(L));
	}
	if(Map.HasSwitch)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_SWITCH);
		L.m_Switch = SwitchData;
		FakeMap.PushItem(MAPITEMTYPE_LAYER, &L, sizeof(L));
	}
	if(Map.HasTune)
	{
		CMapItemLayerTilemap L = MakeTilemapItem((int)Map.Width, (int)Map.Height, TILESLAYERFLAG_TUNE);
		L.m_Tune = TuneData;
		FakeMap.PushItem(MAPITEMTYPE_LAYER, &L, sizeof(L));
	}

	Layers.Init(&FakeMap, false, false);
	Collision.Init(&Layers);
}

// =============================================================================================
// Tuning overrides: bypasses CTuningParams::Set's float round-trip (`(int)(v * 100.0f)`) by
// writing the fixed-point value directly through NetworkArray(), so a scenario's exact
// `value_x100` integer is what the core actually tunes with — no intermediate float rounding.
// =============================================================================================

int TuningIndexByName(const char *Name)
{
	for(int i = 0; i < CTuningParams::Num(); i++)
		if(str_comp_nocase(CTuningParams::Name(i), Name) == 0)
			return i;
	return -1;
}

void ApplyTuningOverrides(CTuningParams &Tuning, const std::vector<TuningOverride> &Overrides)
{
	for(const auto &O : Overrides)
	{
		int Index = TuningIndexByName(O.Name.c_str());
		if(Index < 0)
			Fail("unknown tuning parameter '" + O.Name + "'");
		Tuning.NetworkArray()[Index] = O.ValueX100;
	}
}

// =============================================================================================
// Trace v1 writer (see docs/formats.md and crates/ddai-trace/src/trace.rs)
// =============================================================================================

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

struct StateFields
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

} // namespace

int main(int argc, char **argv)
{
	if(argc < 4)
	{
		fprintf(stderr, "usage: %s <rawmap-file> <scenario-file> <trace-out-file> [--generator NAME --seed N]\n", argv[0]);
		return 1;
	}
	std::string RawmapPath = argv[1];
	std::string ScenarioPath = argv[2];
	std::string TraceOutPath = argv[3];
	std::string GeneratorName;
	bool HasSeed = false;
	uint64_t Seed = 0;
	for(int i = 4; i < argc; i++)
	{
		if(!strcmp(argv[i], "--generator") && i + 1 < argc)
			GeneratorName = argv[++i];
		else if(!strcmp(argv[i], "--seed") && i + 1 < argc)
		{
			Seed = strtoull(argv[++i], nullptr, 10);
			HasSeed = true;
		}
		else
			Fail(std::string("unknown argument: ") + argv[i]);
	}

	std::vector<unsigned char> RawmapBytes = ReadFile(RawmapPath);
	auto RawmapDigest = oracle_sha256::Digest(RawmapBytes);
	RawMap Map = ParseRawMap(RawmapBytes);

	std::vector<unsigned char> ScenarioBytes = ReadFile(ScenarioPath);
	auto ScenarioDigest = oracle_sha256::Digest(ScenarioBytes);
	Scenario S = ParseScenario(ScenarioBytes);

	if(memcmp(RawmapDigest.data(), S.MapSha256.data(), 32) != 0)
	{
		Fail("rawmap sha256 (" + oracle_sha256::ToHex(RawmapDigest) + ") does not match scenario's declared map_sha256 (" +
			oracle_sha256::ToHex(S.MapSha256) + ")");
	}

	CFakeMap FakeMap;
	CLayers Layers;
	CCollision Collision;
	BuildCollision(Map, FakeMap, Layers, Collision);

	const size_t N = S.Characters.size();
	if(N == 0)
		Fail("scenario has no characters");
	if(N > MAX_CLIENTS)
		Fail("scenario has more characters than MAX_CLIENTS");

	CTeamsCore Teams;
	CWorldCore World;
	// `std::vector<T>(N)` value-initializes every element: for a class type (like
	// `CCharacterCore`) whose default constructor is implicit rather than user-provided, that
	// means every member is zero-initialized first. This matters because `CCharacterCore::Reset()`
	// does *not* initialize `m_ActiveWeapon`/`m_Colliding`/`m_LeftWall`/`m_MoveRestrictions`/
	// `m_Id` (see docs/formats.md) — without this, they would be indeterminate. This harness
	// defines their starting value as `0`/`false` by construction.
	std::vector<CCharacterCore> Cores(N);
	CTuningParams Tuning; // CTuningParams::DEFAULT semantics: its constructor sets every field's
	// real default (see CTuningParams(), gamecore.h) — this is a fresh CTuningParams(), not a
	// zero-initialized struct.
	ApplyTuningOverrides(Tuning, S.TuningOverrides);

	for(size_t Slot = 0; Slot < N; Slot++)
	{
		CCharacterCore &Core = Cores[Slot];
		Core.Reset();
		Core.Init(&World, &Collision, &Teams);
		Core.m_Id = (int)S.Characters[Slot].Id;
		Core.m_Pos = vec2((float)S.Characters[Slot].SpawnX, (float)S.Characters[Slot].SpawnY);
		Core.m_Tuning = Tuning;
		World.m_apCharacters[S.Characters[Slot].Id] = &Core;
	}

	// Tick order: newest-spawned-first. Every character in a scenario spawns at tick 0 in
	// ascending slot order (0..N-1), so on a real server each insertion goes to the head of
	// CGameWorld's per-type entity list — the list ends up in slot order N-1, N-2, ..., 0, and
	// (since nothing in this core-only harness ever spawns/removes a character mid-run) stays
	// that way for the rest of the scenario. See docs/formats.md and
	// docs/research/ddnet-physics.md §2/§3.B2 for the server behavior this mirrors.
	std::vector<size_t> TickOrder(N);
	for(size_t i = 0; i < N; i++)
		TickOrder[i] = N - 1 - i;

	std::vector<int32_t> CapturedMoveRestrictions(N, 0);
	std::vector<std::vector<StateFields>> AllStates(S.Inputs.size(), std::vector<StateFields>(N));
	// The *resolved* input actually applied each tick — what the trace's "input applied" record
	// must be (review round 1, finding F3: the trace records the applied input, not the stored
	// aim-mode recipe).
	std::vector<std::vector<ResolvedInput>> AllResolvedInputs(S.Inputs.size(), std::vector<ResolvedInput>(N));

	// `resolve_input`'s "previous tick's post-Quantize() integer position" (review round 1,
	// finding F3) — at tick 0 there is no previous tick, so it starts at each character's spawn
	// position, exactly as `crates/ddai-trace/src/scenario.rs`'s `Scenario::spawn_positions`
	// documents. Updated after every tick's Move()+Quantize() below.
	std::vector<std::pair<int32_t, int32_t>> PrevPositions(N);
	for(size_t Slot = 0; Slot < N; Slot++)
		PrevPositions[Slot] = {S.Characters[Slot].SpawnX, S.Characters[Slot].SpawnY};

	auto Start = std::chrono::steady_clock::now();
	for(size_t Tick = 0; Tick < S.Inputs.size(); Tick++)
	{
		for(size_t Slot = 0; Slot < N; Slot++)
		{
			ResolvedInput In = ResolveInput(S.Inputs[Tick][Slot], Slot, PrevPositions);
			CNetObj_PlayerInput NetIn{};
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
			Cores[Slot].m_Input = NetIn;
			AllResolvedInputs[Tick][Slot] = In;
		}

		// Mirrors CGameWorld::Tick (gameworld.cpp) restricted to CCharacter::Tick's core calls
		// (character.cpp:815 `m_Core.Tick(true, !SvNoWeakHook)`) — see docs/formats.md for the
		// derivation of this order from the server source.
		if(S.NoWeakHook)
		{
			for(size_t Slot : TickOrder)
				Cores[Slot].Tick(true, false);
			for(size_t Slot : TickOrder)
				Cores[Slot].TickDeferred();
		}
		else
		{
			for(size_t Slot : TickOrder)
				Cores[Slot].Tick(true, true);
		}

		// Capture the move-restrictions value each core computed at the start of its Tick() —
		// `m_MoveRestrictions` is private with no getter, but Tick() never mutates `m_Pos`
		// before this point, so recomputing the identical (pure) query now reproduces the exact
		// value Tick() used. See docs/formats.md.
		for(size_t Slot = 0; Slot < N; Slot++)
			CapturedMoveRestrictions[Slot] = Collision.GetMoveRestrictions(Cores[Slot].m_Pos);

		for(size_t Slot : TickOrder)
		{
			Cores[Slot].Move();
			Cores[Slot].Quantize();
		}

		// `m_Pos` is exactly integer-valued after `Quantize()` — safe to truncate to `int32_t`
		// with no rounding — and becomes the *next* tick's `resolve_input` "previous position".
		for(size_t Slot = 0; Slot < N; Slot++)
			PrevPositions[Slot] = {(int32_t)Cores[Slot].m_Pos.x, (int32_t)Cores[Slot].m_Pos.y};

		for(size_t Slot = 0; Slot < N; Slot++)
		{
			const CCharacterCore &Core = Cores[Slot];
			StateFields &St = AllStates[Tick][Slot];
			St.PosX = Core.m_Pos.x;
			St.PosY = Core.m_Pos.y;
			St.VelX = Core.m_Vel.x;
			St.VelY = Core.m_Vel.y;
			St.HookPosX = Core.m_HookPos.x;
			St.HookPosY = Core.m_HookPos.y;
			St.HookDirX = Core.m_HookDir.x;
			St.HookDirY = Core.m_HookDir.y;
			St.HookTeleBaseX = Core.m_HookTeleBase.x;
			St.HookTeleBaseY = Core.m_HookTeleBase.y;
			St.HookTick = Core.m_HookTick;
			St.HookState = Core.m_HookState;
			St.HookedPlayer = Core.HookedPlayer();
			St.ActiveWeapon = Core.m_ActiveWeapon;
			St.NewHook = Core.m_NewHook ? 1 : 0;
			St.Jumped = Core.m_Jumped;
			St.JumpedTotal = Core.m_JumpedTotal;
			St.Jumps = Core.m_Jumps;
			St.Direction = Core.m_Direction;
			St.Angle = Core.m_Angle;
			St.TriggeredEvents = Core.m_TriggeredEvents;
			St.Colliding = Core.m_Colliding;
			St.LeftWall = Core.m_LeftWall ? 1 : 0;
			St.MoveRestrictions = CapturedMoveRestrictions[Slot];
			St.Solo = Core.m_Solo ? 1 : 0;
			St.CollisionDisabled = Core.m_CollisionDisabled ? 1 : 0;
			St.EndlessHook = Core.m_EndlessHook ? 1 : 0;
			St.HookHitDisabled = Core.m_HookHitDisabled ? 1 : 0;
		}
	}
	auto End = std::chrono::steady_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();
	double TeeTicks = (double)S.Inputs.size() * (double)N;
	double Throughput = Seconds > 0 ? TeeTicks / Seconds : 0.0;

	// --- Metadata JSON -------------------------------------------------------------------------
	std::ostringstream Json;
	Json << "{";
	Json << "\"producer\":{\"name\":\"ddnet-oracle-a\",\"version\":\"1\"},";
	Json << "\"ddnet\":{\"tag\":\"20.1\",\"commit\":\"c9d208138f85755521f16a0096b6fe036c5c8698\"},";
	Json << "\"map_sha256\":\"" << oracle_sha256::ToHex(RawmapDigest) << "\",";
	Json << "\"scenario\":{";
	if(!GeneratorName.empty())
		Json << "\"generator\":\"" << JsonEscape(GeneratorName) << "\",";
	if(HasSeed)
		Json << "\"seed\":" << Seed << ",";
	Json << "\"scenario_sha256\":\"" << oracle_sha256::ToHex(ScenarioDigest) << "\"";
	Json << "},";
	Json << "\"input_schema\":[[\"direction\",\"i32\"],[\"target_x\",\"i32\"],[\"target_y\",\"i32\"],[\"jump\",\"i32\"],"
			"[\"fire\",\"i32\"],[\"hook\",\"i32\"],[\"player_flags\",\"i32\"],[\"wanted_weapon\",\"i32\"],"
			"[\"next_weapon\",\"i32\"],[\"prev_weapon\",\"i32\"]],";
	Json << "\"state_schema\":[[\"pos_x\",\"f32\"],[\"pos_y\",\"f32\"],[\"vel_x\",\"f32\"],[\"vel_y\",\"f32\"],"
			"[\"hook_pos_x\",\"f32\"],[\"hook_pos_y\",\"f32\"],[\"hook_dir_x\",\"f32\"],[\"hook_dir_y\",\"f32\"],"
			"[\"hook_tele_base_x\",\"f32\"],[\"hook_tele_base_y\",\"f32\"],[\"hook_tick\",\"i32\"],"
			"[\"hook_state\",\"i32\"],[\"hooked_player\",\"i32\"],[\"active_weapon\",\"i32\"],[\"new_hook\",\"i32\"],"
			"[\"jumped\",\"i32\"],[\"jumped_total\",\"i32\"],[\"jumps\",\"i32\"],[\"direction\",\"i32\"],"
			"[\"angle\",\"i32\"],[\"triggered_events\",\"i32\"],[\"colliding\",\"i32\"],[\"left_wall\",\"i32\"],"
			"[\"move_restrictions\",\"i32\"],[\"solo\",\"i32\"],[\"collision_disabled\",\"i32\"],"
			"[\"endless_hook\",\"i32\"],[\"hook_hit_disabled\",\"i32\"]]";
	Json << "}";

	// --- Trace v1 body --------------------------------------------------------------------------
	ByteWriter W;
	W.Magic("TRC1");
	W.U32(1);
	W.String32(Json.str());
	W.U32((uint32_t)N);
	for(size_t Slot = 0; Slot < N; Slot++)
		W.U32(S.Characters[Slot].Id);
	W.U32((uint32_t)S.Inputs.size());
	for(size_t Tick = 0; Tick < S.Inputs.size(); Tick++)
	{
		for(size_t Slot = 0; Slot < N; Slot++)
		{
			const ResolvedInput &In = AllResolvedInputs[Tick][Slot];
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
			AllStates[Tick][Slot].Write(W);
		}
	}
	WriteFile(TraceOutPath, W.Data());

	fprintf(stderr, "oracle_core: %zu ticks, %zu characters, %.0f tee-ticks/s, wrote %s (%zu bytes)\n", S.Inputs.size(), N,
		Throughput, TraceOutPath.c_str(), W.Data().size());
	return 0;
}
