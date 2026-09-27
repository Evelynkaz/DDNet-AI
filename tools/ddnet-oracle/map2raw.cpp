// map2raw: reads a real DDNet .map file with DDNet 20.1's OWN unmodified map-loading code
// (`engine/shared/datafile.cpp` + `engine/shared/map.cpp` + `game/layers.cpp`, compiled from the
// sources fetch.sh fetches into build/, never committed to this repository) and writes its
// physics-relevant layers (game/front/tele/speedup/switch/tune tiles + map Settings) as a rawmap
// v1 file (docs/formats.md §1) — the same format `ddai-map::load_map` produces from the same
// input. `tools/ddnet-oracle/map-corpus-check.sh` diffs the two outputs byte-for-byte across the
// real map corpus; that is this tool's only purpose (task 1.4 in docs/PLAN.md).
//
// This file is DDNet-AI's own original code (GPL-3.0-only, like the rest of this repo); no DDNet
// source is copied into it — it only calls the real, compiled DDNet functions through their
// public headers, the same way `oracle_core.cpp` (task 1.2) calls `CCharacterCore`/`CCollision`.
//
// What this tool replicates from real DDNet loading (see file:line comments below, all against
// the DDNet 20.1 tag `c9d208138f85755521f16a0096b6fe036c5c8698` sources fetch.sh pins):
//   - `CMap::Load` (engine/shared/map.cpp): datafile version check, MAPITEMTYPE_VERSION == 1,
//     group/layer traversal picking the LAST tilemap layer with each physics flag (no `break` in
//     the loop — see layers.cpp:22-77), tilemap item version upgrade (2/3/4, legacy DDRace
//     layouts), width/height >= 2, at-most-one-physics-flag, tile-skip unpacking (version >= 4),
//     zero-padding checks, data-index uniqueness, physics-layer->=game-size check.
//   - `CLayers::Init(pMap, /*GameOnly=*/false, /*InitializeTilemapSkip=*/false)` — exactly what
//     the DDNet SERVER calls (game/server/gamecontext.cpp:4105), not the client/editor (which
//     pass `InitializeTilemapSkip=true`, a purely cosmetic render-batching hint this tool never
//     needs — see docs/DECISIONS.md-style note in ddai-map's loader.rs for the same choice).
//   - `CGameContext::LoadMapSettings` (game/server/gamecontext.cpp:4556-4580): the *only* real
//     server code path that reads the map's Settings blob — finds the first MAPITEMTYPE_INFO
//     item with Id 0, and if it is large enough and `m_Settings > -1`, splits that data blob on
//     NUL bytes. Because this calls the real `IMap::GetData`, it also naturally reproduces a
//     genuine DDNet quirk (review round 1 finding F6): `CDatafile::GetData` caches one
//     decompressed-data *processor* per data index, shared by every consumer of that index — if
//     some tiles layer's own data happens to share the exact index the Settings blob lives at,
//     that layer's content-validation processor (registered by `CMap::Load` for every tiles
//     layer, physics or decorative) runs first and typically rejects the blob as "too small to be
//     this layer's tile data", so `GetData` returns `nullptr` to *this* settings read too, even
//     though nothing is wrong with the Settings blob itself. No fix needed on this side — calling
//     the real, unmodified `GetData` already reproduces this exactly; only `ddai-map`'s Rust port
//     (which doesn't have a shared per-index cache to fall out of naturally) needed a matching
//     special case (`crate::loader::read_info`'s `used_by_tiles_layer` check).
//   - `CCollision::Init` (collision.cpp:59-86): every physics layer *other than game* is loaded
//     lazily and null-tolerantly — a bad blob (wrong padding, a disallowed tile-skip role,
//     truncated, corrupt, or a declared/actual size mismatch) leaves that one layer absent, not
//     the whole map rejected (`GetOptionalPhysicsLayerData`, below). The GAME layer is the one
//     exception: `CMap::Load` itself force-loads it eagerly (map.cpp:190-195), so a failure there
//     really does rejects the whole map (`GetGameLayerData`, `Fail()`s).
//
// What this tool deliberately does NOT replicate (see docs/formats.md and the build report for
// task 1.4): `MapItemInfo` (author/version/credits/license) — no server code path reads these at
// all (only the editor's `CEditorMap::Load` does, and rawmap v1 has no field for them anyway;
// `ddai-map`'s Rust `MapInfo` is verified by Rust-only unit tests, not cross-checked here).
//
// Usage: map2raw <input.map> <output.rawmap>

#include <base/hash.h>
#include <base/io.h>
#include <base/log.h>
#include <base/types.h>

#include <engine/kernel.h>
#include <engine/map.h>
#include <engine/shared/config.h>
#include <engine/shared/datafile.h>
#include <engine/shared/map.h>
#include <engine/storage.h>

#include <game/layers.h>
#include <game/mapitems.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// -------------------------------------------------------------------------------------------
// Stub externs. This tool links the REAL `engine/shared/{datafile,map}.cpp`, `game/layers.cpp`
// (plus `game/{gamecore,collision,teamscore,prng}.cpp`, only pulled in transitively because
// `layers.cpp`'s callers link against the same object set as `oracle_core.cpp` — see build.sh)
// and the REAL `base/{str,mem,io,hash_libtomcrypt}.cpp` (so `IntsToStr`/`str_comp`/UTF-8
// checks/etc. are DDNet's actual implementations, not reimplementations). Only the handful of
// free functions below are stubbed, exactly like `oracle_core.cpp` already stubs
// `dbg_assert_imp`: each is either (a) never reachable for the built-in, non-UUID map item types
// our maps use (`g_UuidManager`/`CUuid`, see datafile.cpp:863-917 `Get{External,Internal}ItemType`
// — every item type this tool cares about is < `OFFSET_UUID_TYPE`/`OFFSET_UUID`, so the
// `g_UuidManager` branch never executes), (b) a pure diagnostic side effect we don't need
// (`log_log`/`log_log_color` — DDNet's own `log_error` etc. are just extra stderr noise here;
// this tool reports failure via `CMap::Load`'s bool return, not by parsing log output), or
// (c) a tiny, self-contained path utility we reimplement directly instead of linking the much
// larger `base/fs.cpp` for two functions (`fs_filename`/`fs_split_file_extension`, both copied
// field-for-field from `base/fs.cpp:461-491` — only used for `CMap`'s cosmetic `BaseName()`, not
// for anything that affects map DATA).
// -------------------------------------------------------------------------------------------

extern "C" void dbg_assert_imp(const char *filename, int line, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "assert %s:%d: ", filename, line);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	abort();
}

void log_log(LEVEL, const char *, const char *, ...) {}
void log_log_color(LEVEL, LOG_COLOR, const char *, const char *, ...) {}

// `game/collision.cpp`/`teamscore.cpp` read `g_Config.m_Sv*` fields (server config, e.g.
// `sv_team`/`sv_old_teleport_hook`) — never relevant to a read-only map loader, but the symbol
// must exist for the same reason `g_UuidManager` above does. `CConfig`'s fields default to 0
// (see `engine/shared/config_variables.h`'s `MACRO_CONFIG_*` expansions), matching DDNet's own
// zero-initialized default before `ConfigManager()->Init()` ever runs — exactly like
// `oracle_core.cpp` (task 1.2) already does for the same reason.
CConfig g_Config;

// base/io.cpp's (unused) `io_current_exe()` calls this; never reached by a read-only loader.
int fs_executable_path(char *, int) { return -1; }

// base/fs.cpp:461-469 — kept identical (path/filename decomposition, not map data).
const char *fs_filename(const char *path)
{
	for(const char *filename = path + strlen(path); filename >= path; --filename)
	{
		if(filename[0] == '/' || filename[0] == '\\')
			return filename + 1;
	}
	return path;
}

// base/fs.cpp:471-... simplified to the 3-arg call `CMap::Load` actually uses (name only, no
// extension output) — `CMap::Load(IStorage*, pPath, Type)` calls
// `fs_split_file_extension(fs_filename(pPath), aFilename, sizeof(aFilename))`.
void fs_split_file_extension(const char *filename, char *name, size_t name_size, char * = nullptr, size_t = 0)
{
	const char *last_dot = strrchr(filename, '.');
	size_t n = (last_dot == nullptr || last_dot == filename) ? strlen(filename) : (size_t)(last_dot - filename);
	if(n >= name_size)
		n = name_size - 1;
	memcpy(name, filename, n);
	name[n] = '\0';
}

// engine/shared/uuid_manager.h — stubbed rather than linking uuid_manager.cpp (which would pull
// in engine/shared/packer.cpp + base/secure.cpp for RandomUuid, never used here). Every map item
// type this tool reads (MAPITEMTYPE_VERSION/INFO/GROUP/LAYER, all < 8) is far below
// `OFFSET_UUID_TYPE` (0x8000) / `OFFSET_UUID` (0x10000), so datafile.cpp:863-917's
// `Get{External,Internal}ItemType` always takes their early-return fast path and never actually
// calls into `g_UuidManager` — these bodies are unreachable for any item this tool cares about.
const CUuid UUID_ZEROED = {};
bool CUuid::operator==(const CUuid &Other) const { return memcmp(m_aData, Other.m_aData, sizeof(m_aData)) == 0; }
bool CUuid::operator!=(const CUuid &Other) const { return !(*this == Other); }
bool CUuid::operator<(const CUuid &Other) const { return memcmp(m_aData, Other.m_aData, sizeof(m_aData)) < 0; }
CUuidManager g_UuidManager;
CUuid CUuidManager::GetUuid(int) const { return UUID_ZEROED; }
int CUuidManager::LookupUuid(CUuid) const { return -1; /* UUID_UNKNOWN */ }

namespace
{

[[noreturn]] void Fail(const std::string &Msg)
{
	fprintf(stderr, "map2raw: %s\n", Msg.c_str());
	exit(1);
}

// -------------------------------------------------------------------------------------------
// Minimal `IStorage`: `CDataFileReader::Open` only ever calls `OpenFile` on the instance we give
// it (engine/shared/datafile.cpp:522) — every other `IStorage` method below is unreachable for a
// read-only `CMap::Load`, so each is a trivial stub. `OpenFile` ignores `Type`/`pBuffer` and just
// opens `pFilename` as a real filesystem path via the real `io_open` (base/io.cpp) — "stub
// storage by opening a file path directly", per the task spec.
// -------------------------------------------------------------------------------------------
class CDirectFileStorage : public IStorage
{
public:
	int NumPaths() const override { return 1; }
	void ListDirectory(int, const char *, FS_LISTDIR_CALLBACK, void *) override {}
	void ListDirectoryInfo(int, const char *, FS_LISTDIR_CALLBACK_FILEINFO, void *) override {}
	IOHANDLE OpenFile(const char *pFilename, int Flags, int, char * = nullptr, int = 0) override
	{
		return io_open(pFilename, Flags);
	}
	bool FileExists(const char *, int) override { return false; }
	bool FolderExists(const char *, int) override { return false; }
	bool ReadFile(const char *, int, void **, unsigned *) override { return false; }
	char *ReadFileStr(const char *, int) override { return nullptr; }
	bool RetrieveTimes(const char *, int, time_t *, time_t *) override { return false; }
	bool CalculateHashes(const char *, int, SHA256_DIGEST *, unsigned *) override { return false; }
	bool FindFile(const char *, const char *, int, char *, int) override { return false; }
	size_t FindFiles(const char *, const char *, int, std::set<std::string> *) override { return 0; }
	bool RemoveFile(const char *, int) override { return false; }
	bool RemoveFolder(const char *, int) override { return false; }
	bool RenameFile(const char *, const char *, int) override { return false; }
	bool CreateFolder(const char *, int) override { return false; }
	void GetCompletePath(int, const char *, char *pBuffer, unsigned BufferSize) override
	{
		if(BufferSize > 0)
			pBuffer[0] = '\0';
	}
	bool RemoveBinaryFile(const char *) override { return false; }
	bool RenameBinaryFile(const char *, const char *) override { return false; }
	const char *GetBinaryPath(const char *, char *pBuffer, unsigned BufferSize) override
	{
		if(BufferSize > 0)
			pBuffer[0] = '\0';
		return pBuffer;
	}
	const char *GetBinaryPathAbsolute(const char *, char *pBuffer, unsigned BufferSize) override
	{
		if(BufferSize > 0)
			pBuffer[0] = '\0';
		return pBuffer;
	}
};

// =============================================================================================
// Tiny little-endian byte writer — mirrors crates/ddai-trace/src/io.rs's `Writer` field-for-field
// (this tool never reads rawmap files, only writes them, so no matching reader is needed here).
// =============================================================================================
class ByteWriter
{
public:
	void U8(uint8_t V) { m_Buf.push_back(V); }
	void U32(uint32_t V)
	{
		for(int i = 0; i < 4; i++)
			m_Buf.push_back((unsigned char)((V >> (8 * i)) & 0xff));
	}
	void I16(int16_t V)
	{
		uint16_t U = (uint16_t)V;
		m_Buf.push_back((unsigned char)(U & 0xff));
		m_Buf.push_back((unsigned char)((U >> 8) & 0xff));
	}
	void Bytes(const void *P, size_t N)
	{
		const unsigned char *B = (const unsigned char *)P;
		m_Buf.insert(m_Buf.end(), B, B + N);
	}
	void Magic(const char M[4]) { Bytes(M, 4); }
	void String32(const char *Data, size_t Len)
	{
		U32((uint32_t)Len);
		Bytes(Data, Len);
	}
	const std::vector<unsigned char> &Data() const { return m_Buf; }

private:
	std::vector<unsigned char> m_Buf;
};

void WriteTiles(ByteWriter &W, const CTile *pTiles, size_t N)
{
	for(size_t i = 0; i < N; i++)
	{
		W.U8(pTiles[i].m_Index);
		W.U8(pTiles[i].m_Flags);
		W.U8(pTiles[i].m_Skip);
		W.U8(pTiles[i].m_MustBe0);
	}
}

// Reads the GAME layer's data blob and returns a pointer to (at least) `N` records, or fails the
// whole tool.
//
// Review round 1 finding F3: an earlier version of this file used one function, ending in a hard
// `Fail()`, for *every* physics layer, including the non-game ones — wrong. The GAME layer's data
// is force-loaded eagerly by `CMap::Load` itself
// (map.cpp:190-195: `if(NewDataFile.GetData(pGameLayer->m_Data) == nullptr) return false;`), so a
// failure there really does mean DDNet's own loader would have already rejected the whole map —
// `Fail()` (a hard tool error) correctly mirrors that. Every *other* physics layer's data is
// loaded lazily by `CCollision::Init` (collision.cpp:59-86), which just stores whatever `GetData`
// returns — including `nullptr` on failure — into `m_pFront`/`m_pTele`/etc. and moves on; nothing
// else in `CMap::Load` forces that call, so a bad front/tele/speedup/switch/tune blob (wrong
// padding, a disallowed tile-skip role, truncated, corrupt, or too small — see `GetOptionalPhysicsLayerData`
// below) never fails the map, only leaves that one layer absent. `Fail()`ing there was this
// tool's own bug (it mirrored this Rust port's earlier, likewise-wrong choice, rather than real
// DDNet), reproduced directly against a mutated real map (`front_badpad.map` in the build report).
template<typename T>
const T *GetGameLayerData(IMap &Map, int DataIndex, size_t N)
{
	if(DataIndex < 0)
		Fail("game layer is present but has no data index");
	const T *pData = static_cast<const T *>(Map.GetData(DataIndex));
	if(pData == nullptr)
		Fail("game layer data failed to load/decompress");
	const size_t GotRecords = (size_t)Map.GetDataSize(DataIndex) / sizeof(T);
	if(GotRecords < N)
		Fail("game layer data is smaller than declared");
	return pData;
}

// For every physics layer *other* than game: matches `CCollision::Init`'s own lazy, null-tolerant
// use of `GetData` (collision.cpp:59-86) — a failure here means "this layer is absent", not "the
// map failed to load", so this returns `nullptr` instead of calling `Fail()`. `GotRecords < N`
// (declared smaller than the game layer) is structurally unreachable for any map `CMap::Load`
// actually accepted (map.cpp:640-648 already requires every physics layer's *declared*
// `width*height` to be `>=` the game layer's) — kept anyway, defensively, in case that invariant
// is ever violated by a future change on either side of the parity comparison.
template<typename T>
const T *GetOptionalPhysicsLayerData(IMap &Map, int DataIndex, size_t N)
{
	if(DataIndex < 0)
		return nullptr;
	const T *pData = static_cast<const T *>(Map.GetData(DataIndex));
	if(pData == nullptr)
		return nullptr;
	const size_t GotRecords = (size_t)Map.GetDataSize(DataIndex) / sizeof(T);
	if(GotRecords < N)
		return nullptr;
	return pData;
}

// Mirrors `CGameContext::LoadMapSettings` (game/server/gamecontext.cpp:4556-4580) — the only real
// DDNet server code path that reads a map's Settings blob. Finds the first MAPITEMTYPE_INFO item
// with Id 0; if it's too small or has no Settings data index, the map simply has zero settings
// (not an error — matches the server exactly: `break` with no error either way).
std::vector<std::string> ReadSettings(IMap &Map)
{
	std::vector<std::string> Settings;
	int Start, Num;
	Map.GetType(MAPITEMTYPE_INFO, &Start, &Num);
	for(int i = Start; i < Start + Num; i++)
	{
		int ItemId;
		auto *pItem = static_cast<CMapItemInfoSettings *>(Map.GetItem(i, nullptr, &ItemId));
		int ItemSize = Map.GetItemSize(i);
		if(!pItem || ItemId != 0)
			continue;
		if(ItemSize < (int)sizeof(CMapItemInfoSettings))
			break;
		if(!(pItem->m_Settings > -1))
			break;
		int Size = Map.GetDataSize(pItem->m_Settings);
		const char *pSettings = static_cast<const char *>(Map.GetData(pItem->m_Settings));
		if(pSettings == nullptr)
			break;
		const char *pNext = pSettings;
		while(pNext < pSettings + Size)
		{
			size_t StrSize = strlen(pNext) + 1;
			Settings.emplace_back(pNext, strlen(pNext));
			pNext += StrSize;
		}
		break;
	}
	return Settings;
}

} // namespace

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		fprintf(stderr, "usage: %s <input.map> <output.rawmap>\n", argv[0]);
		return 2;
	}
	const char *pInputPath = argv[1];
	const char *pOutputPath = argv[2];

	CDirectFileStorage Storage;
	CMap Map;
	// `CMap::Load(IStorage*, pPath, Type)` (engine/shared/map.h:40) derives a cosmetic base name
	// via `fs_filename`/`fs_split_file_extension` (stubbed above) then delegates to the 4-arg
	// overload that actually opens+parses the file; `Type` is passed straight to our
	// `CDirectFileStorage::OpenFile`, which ignores it and opens `pInputPath` directly.
	if(!Map.Load(&Storage, pInputPath, IStorage::TYPE_ALL))
	{
		fprintf(stderr, "map2raw: rejected by CMap::Load (see stderr above for the reason DDNet's own loader gave, if log_log weren't stubbed to silence — rerun with a debug build of DDNet to see it)\n");
		return 1;
	}

	CLayers Layers;
	// `false, false` — exactly what the DDNet SERVER passes (game/server/gamecontext.cpp:4105):
	// `GameOnly=false` (we want every physics layer, not just the game layer) and
	// `InitializeTilemapSkip=false` (that flag re-derives `CTile::m_Skip` as a client-rendering
	// batching hint; the server never does this, and `ddai-map`'s Rust loader must match the
	// server, not the client/editor — see docs/formats.md and this tool's header comment).
	Layers.Init(&Map, false, false);

	const CMapItemLayerTilemap *pGameLayer = Layers.GameLayer();
	if(pGameLayer == nullptr)
		Fail("no game layer (should be impossible: CMap::Load requires one)");
	const uint32_t Width = (uint32_t)pGameLayer->m_Width;
	const uint32_t Height = (uint32_t)pGameLayer->m_Height;
	const size_t N = (size_t)Width * (size_t)Height;

	const CTile *pGame = GetGameLayerData<CTile>(Map, pGameLayer->m_Data, N);

	const CMapItemLayerTilemap *pFrontLayer = Layers.FrontLayer();
	const CMapItemLayerTilemap *pTeleLayer = Layers.TeleLayer();
	const CMapItemLayerTilemap *pSpeedupLayer = Layers.SpeedupLayer();
	const CMapItemLayerTilemap *pSwitchLayer = Layers.SwitchLayer();
	const CMapItemLayerTilemap *pTuneLayer = Layers.TuneLayer();

	const CTile *pFront = pFrontLayer ? GetOptionalPhysicsLayerData<CTile>(Map, pFrontLayer->m_Front, N) : nullptr;
	const CTeleTile *pTele = pTeleLayer ? GetOptionalPhysicsLayerData<CTeleTile>(Map, pTeleLayer->m_Tele, N) : nullptr;
	const CSpeedupTile *pSpeedup = pSpeedupLayer ? GetOptionalPhysicsLayerData<CSpeedupTile>(Map, pSpeedupLayer->m_Speedup, N) : nullptr;
	const CSwitchTile *pSwitch = pSwitchLayer ? GetOptionalPhysicsLayerData<CSwitchTile>(Map, pSwitchLayer->m_Switch, N) : nullptr;
	const CTuneTile *pTune = pTuneLayer ? GetOptionalPhysicsLayerData<CTuneTile>(Map, pTuneLayer->m_Tune, N) : nullptr;

	std::vector<std::string> Settings = ReadSettings(Map);

	// --- rawmap v1 (docs/formats.md §1) ---------------------------------------------------------
	uint8_t Present = 0;
	if(pFront)
		Present |= 1 << 0;
	if(pTele)
		Present |= 1 << 1;
	if(pSpeedup)
		Present |= 1 << 2;
	if(pSwitch)
		Present |= 1 << 3;
	if(pTune)
		Present |= 1 << 4;

	ByteWriter W;
	W.Magic("RMP1");
	W.U32(1);
	W.U32(Width);
	W.U32(Height);
	W.U8(Present);
	WriteTiles(W, pGame, N);
	if(pFront)
		WriteTiles(W, pFront, N);
	if(pTele)
	{
		for(size_t i = 0; i < N; i++)
		{
			W.U8(pTele[i].m_Number);
			W.U8(pTele[i].m_Type);
		}
	}
	if(pSpeedup)
	{
		for(size_t i = 0; i < N; i++)
		{
			W.U8(pSpeedup[i].m_Force);
			W.U8(pSpeedup[i].m_MaxSpeed);
			W.U8(pSpeedup[i].m_Type);
			W.U8(0 /* reserved, mirrors CSpeedupTile::m_MustBe0 */);
			W.I16(pSpeedup[i].m_Angle);
		}
	}
	if(pSwitch)
	{
		for(size_t i = 0; i < N; i++)
		{
			W.U8(pSwitch[i].m_Number);
			W.U8(pSwitch[i].m_Type);
			W.U8(pSwitch[i].m_Flags);
			W.U8(pSwitch[i].m_Delay);
		}
	}
	if(pTune)
	{
		for(size_t i = 0; i < N; i++)
		{
			W.U8(pTune[i].m_Number);
			W.U8(pTune[i].m_Type);
		}
	}
	W.U32((uint32_t)Settings.size());
	for(const std::string &S : Settings)
		W.String32(S.data(), S.size());

	std::ofstream Out(pOutputPath, std::ios::binary);
	if(!Out)
		Fail(std::string("cannot open ") + pOutputPath + " for writing");
	Out.write((const char *)W.Data().data(), (std::streamsize)W.Data().size());
	if(!Out)
		Fail(std::string("failed writing ") + pOutputPath);

	fprintf(stderr, "map2raw: %ux%u, present=0x%02x, %zu settings, wrote %s (%zu bytes)\n",
		Width, Height, Present, Settings.size(), pOutputPath, W.Data().size());
	return 0;
}
