// demo2json: reads a real DDNet `.demo` file with DDNet 20.1's OWN unmodified demo-reading code
// (`engine/shared/demo.cpp` + `engine/shared/snapshot.cpp` + `engine/shared/compression.cpp` +
// `engine/shared/huffman.cpp`, compiled from the sources fetch.sh fetches into build/, never
// committed to this repository) and writes the header plus every fully-decoded snapshot's items
// (key + raw int data, per tick) as line-delimited JSON. `parity_check_demo.sh` diffs this
// output against `ddai-demo`'s own decode of the same file, item-for-item — that is this tool's
// only purpose (task 8.4b acceptance criterion 2 in docs/PLAN.md).
//
// This file is DDNet-AI's own original code (GPL-3.0-only, like the rest of this repo); no DDNet
// source is copied into it — it only calls the real, compiled DDNet functions through their
// public headers, the same way `oracle_core.cpp` (task 1.2) and `map2raw.cpp` (task 1.4) do. Also
// links `base/hash.cpp` (real `sha256`/`sha256_str`, needed only because they live in the same
// translation unit as `CDemoPlayer::ExtractMap`/`CDemoRecorder::Start` — never actually called by
// this tool's `main`, same "whole object file" reasoning as everything else here).
//
// What this tool relies on from real DDNet demo playback (see file:line comments below, all
// against the DDNet 20.1 tag `c9d208138f85755521f16a0096b6fe036c5c8698` sources fetch.sh pins):
//   - `CDemoPlayer::Load` (`demo.cpp:842-913`): header/timeline-marker/SHA256-extension parsing,
//     `ScanFile` (`demo.cpp:599-674`) for first/last tick and keyframe positions.
//   - `CDemoPlayer::Play`/`Update` (`demo.cpp:1000-1025,1151-1264`) driving `DoTick`
//     (`demo.cpp:676-822`) exactly the way the only real caller in the codebase does
//     (`CDemoEditor::Slice`, `demo.cpp:1489-1499`): `Play()` once, then `Update(/*RealTime=*/
//     false)` in a loop until paused — see this tool's `main` for why a single `Update(false)`
//     call already drains the whole file (its own internal loop only stops at `RealTime &&
//     ...`, never true here).
//   - `CDemoPlayer::IListener::OnDemoPlayerSnapshot` (`demo.h:74`): called with the fully
//     decoded, delta-applied `CSnapshot*` for the listener's current tick
//     (`Info()->m_Info.m_CurrentTick`) — this tool's `CJsonListener` dumps exactly that snapshot's
//     items (`CSnapshot::NumItems`/`GetItem`/`GetItemSize`, `snapshot.cpp:23-35`), nothing more:
//     no UUID/ex-type resolution (this tool never calls `GetExternalItemType`/`FindItem`, so
//     `g_UuidManager` below is a trivial, unreachable stub, exactly like `map2raw.cpp`'s), no
//     message decoding (`OnDemoPlayerMessage` is a no-op — task acceptance criterion 2's
//     exactness check is scoped to snapshot items only, matching this crate's build report).
//   - `CSnapshotDelta::SetStaticsize` (`snapshot.cpp:289-294`) for the 20 numbered 0.6+DDNet
//     object/event types, set here as `sizeof(CNetObj_*)`/`sizeof(CNetEvent_*)` directly from
//     `generated/protocol.h` (fetch.sh's `network_header` target) — the exact same values real
//     DDNet computes via `CNetObjHandler::GetObjSize` (`datasrc/compile.py`'s `ms_aObjSizes`,
//     `sizeof(CNetObj_X)` for each), reproduced this way (rather than linking the much larger
//     generated `CNetObjHandler`/message-pack machinery this tool never otherwise needs) since
//     `CGameClient::OnInit`'s own setup loop (`gameclient.cpp:358-360`) does exactly this for
//     `i` in `0..NUM_NETOBJTYPES`; index 0 (`NETOBJTYPE_EX`) has no fixed struct/size and is
//     skipped, matching `GetObjSize(0) == 0` in the real table.
//
// Usage: demo2json <input.demo> <output.jsonl>

#include <base/hash.h>
#include <base/io.h>
#include <base/log.h>

#include <engine/demo.h>
#include <engine/shared/config.h>
#include <engine/shared/demo.h>
#include <engine/shared/huffman.h>
#include <engine/shared/network.h>
#include <engine/shared/snapshot.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <string>

// -------------------------------------------------------------------------------------------
// Stub externs — same rationale and same stubs as `map2raw.cpp` (task 1.4): this tool links the
// REAL `engine/shared/{demo,snapshot,compression,huffman}.cpp` and REAL
// `base/{str,mem,io,time,hash_libtomcrypt,bytes}.cpp` (+ `base/unicode/{tolower,tolower_data}.cpp`,
// pulled in transitively by `str.cpp`), so every actual demo-format/compression/UTF-8-validation
// behavior is DDNet's own; only the handful of free functions below are stubbed, each either (a)
// unreachable for anything this tool's `main` calls (`g_UuidManager`/`CUuid` — see the header
// comment above), (b) a pure diagnostic side effect this tool doesn't need (`log_log`/
// `log_log_color`; a failed `Load`/parse is reported via return values, not log output), or (c)
// a tiny, self-contained path utility reimplemented directly instead of linking the larger
// `base/fs.cpp` for two functions (`fs_filename`/`fs_split_file_extension`, copied field-for-
// field from `base/fs.cpp:461-491`, exactly like `map2raw.cpp` — only used for
// `CDemoPlayer::GetDemoName`, never called by this tool, but still compiled into the same
// translation unit as everything this tool does call).
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

// `CSnapshot::DebugDump`/`CSnapshotDelta::DebugDumpDelta` (unreachable — this tool never calls
// them, only `NumItems`/`GetItem`/`GetItemSize`) reference `dbg_msg`; stubbed rather than linking
// `base/dbg.cpp` for the same reason `dbg_assert_imp` above is hand-written, not linked.
void dbg_msg(const char *, const char *, ...) {}

// `CNetBase::Compress`/`Decompress` (`network.cpp:534-542` in the real tree) are what
// `CDemoRecorder::Write`/`CDemoPlayer::DoTick` actually call for the Huffman stage
// (`demo.cpp:303,723`) — defined here directly against this tool's own `CHuffman` instance
// (`engine/shared/huffman.cpp`, linked) instead of linking the much larger `network.cpp` (whose
// own `CNetBase::ms_Huffman` these two static methods would otherwise use): a member function's
// *definition* needs no access to its class's other members, so this is a complete, correct
// implementation of exactly the two static methods demo.cpp calls, nothing else of `CNetBase`.
int CNetBase::Compress(const void *pData, int DataSize, void *pOutput, int OutputSize)
{
	static CHuffman s_Huffman;
	static bool s_Init = false;
	if(!s_Init)
	{
		s_Huffman.Init();
		s_Init = true;
	}
	return s_Huffman.Compress(pData, DataSize, pOutput, OutputSize);
}
int CNetBase::Decompress(const void *pData, int DataSize, void *pOutput, int OutputSize)
{
	static CHuffman s_Huffman;
	static bool s_Init = false;
	if(!s_Init)
	{
		s_Huffman.Init();
		s_Init = true;
	}
	return s_Huffman.Decompress(pData, DataSize, pOutput, OutputSize);
}

// `demo.cpp:828,837,908-909` reads `g_Config.m_ClVideoPauseWithDemo`/`m_ClDemoSliceBegin`/
// `m_ClDemoSliceEnd` — never relevant to a read-only, non-video-recording demo player, but the
// symbol must exist. Zero-initialized, matching DDNet's own default before
// `ConfigManager()->Init()` ever runs — exactly like `oracle_core.cpp`/`map2raw.cpp`.
CConfig g_Config;

// base/io.cpp's (unused here) `io_current_exe()` calls this; never reached by this tool.
int fs_executable_path(char *, int) { return -1; }

// base/fs.cpp:461-469 — kept identical (path/filename decomposition, not demo data).
const char *fs_filename(const char *path)
{
	for(const char *filename = path + strlen(path); filename >= path; --filename)
	{
		if(filename[0] == '/' || filename[0] == '\\')
			return filename + 1;
	}
	return path;
}

// base/fs.cpp:471-... simplified to the 3-arg call `CDemoPlayer::GetDemoName` actually uses.
void fs_split_file_extension(const char *filename, char *name, size_t name_size, char * = nullptr, size_t = 0)
{
	const char *last_dot = strrchr(filename, '.');
	size_t n = (last_dot == nullptr || last_dot == filename) ? strlen(filename) : (size_t)(last_dot - filename);
	if(n >= name_size)
		n = name_size - 1;
	memcpy(name, filename, n);
	name[n] = '\0';
}

// engine/shared/uuid_manager.h — stubbed rather than linking uuid_manager.cpp, exactly like
// `map2raw.cpp`: this tool never calls `CSnapshot::GetExternalItemType`/`FindItem` (see the
// header comment above), so `g_UuidManager` is unreachable for anything this tool does.
const CUuid UUID_ZEROED = {};
bool CUuid::operator==(const CUuid &Other) const { return memcmp(m_aData, Other.m_aData, sizeof(m_aData)) == 0; }
bool CUuid::operator!=(const CUuid &Other) const { return !(*this == Other); }
bool CUuid::operator<(const CUuid &Other) const { return memcmp(m_aData, Other.m_aData, sizeof(m_aData)) < 0; }
CUuidManager g_UuidManager;
CUuid CUuidManager::GetUuid(int) const { return UUID_ZEROED; }
int CUuidManager::LookupUuid(CUuid) const { return -1; /* UUID_UNKNOWN */ }

namespace
{

// -------------------------------------------------------------------------------------------
// Minimal `IStorage`, field-for-field the same as `map2raw.cpp`'s `CDirectFileStorage`:
// `CDemoPlayer::Load`/`GetDemoInfo` only ever call `OpenFile` on the instance we give them
// (`demo.cpp:1324`) — every other `IStorage` method is unreachable, so each is a trivial stub.
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

[[noreturn]] void Fail(const std::string &Msg)
{
	fprintf(stderr, "demo2json: %s\n", Msg.c_str());
	exit(1);
}

// Writes `pStr` as a JSON string literal (with quotes), escaping the handful of bytes JSON
// requires escaped. Demo header strings are already validated UTF-8 C strings
// (`CDemoHeader::Valid`, `demo.cpp:39-47`) with no embedded NULs, so nothing fancier is needed.
void JsonString(FILE *pOut, const char *pStr)
{
	fputc('"', pOut);
	for(const unsigned char *p = (const unsigned char *)pStr; *p; p++)
	{
		switch(*p)
		{
		case '"': fputs("\\\"", pOut); break;
		case '\\': fputs("\\\\", pOut); break;
		case '\n': fputs("\\n", pOut); break;
		case '\r': fputs("\\r", pOut); break;
		case '\t': fputs("\\t", pOut); break;
		default:
			if(*p < 0x20)
				fprintf(pOut, "\\u%04x", *p);
			else
				fputc(*p, pOut);
		}
	}
	fputc('"', pOut);
}

// `CDemoPlayer::IListener` (`demo.h:70-76`): dumps every fully-decoded snapshot's items as one
// JSON line per `OnDemoPlayerSnapshot` call — see the header comment for exactly what/why.
class CJsonListener : public CDemoPlayer::IListener
{
public:
	CDemoPlayer *m_pPlayer = nullptr;
	FILE *m_pOut = nullptr;

	void OnDemoPlayerSnapshot(void *pData, int) override
	{
		const CSnapshot *pSnap = (const CSnapshot *)pData;
		const int Tick = m_pPlayer->Info()->m_Info.m_CurrentTick;
		fprintf(m_pOut, "{\"tick\":%d,\"items\":[", Tick);
		for(int i = 0; i < pSnap->NumItems(); i++)
		{
			if(i > 0)
				fputc(',', m_pOut);
			const CSnapshotItem *pItem = pSnap->GetItem(i);
			const int ItemSize = pSnap->GetItemSize(i); // bytes, excluding the key int itself
			fprintf(m_pOut, "{\"key\":%d,\"data\":[", pItem->Key());
			const int NumInts = ItemSize / (int)sizeof(int32_t);
			for(int b = 0; b < NumInts; b++)
			{
				if(b > 0)
					fputc(',', m_pOut);
				fprintf(m_pOut, "%d", pItem->Data()[b]);
			}
			fputs("]}", m_pOut);
		}
		fputs("]}\n", m_pOut);
	}

	void OnDemoPlayerMessage(void *, int) override {} // out of scope — see the header comment.
};

} // namespace

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		fprintf(stderr, "usage: %s <input.demo> <output.jsonl>\n", argv[0]);
		return 2;
	}
	const char *pInputPath = argv[1];
	const char *pOutputPath = argv[2];

	CDirectFileStorage Storage;

	// `CGameClient::OnInit`'s static-size setup (`gameclient.cpp:358-360`) — see the header
	// comment for why this tool reproduces it via `sizeof(CNetObj_*)` directly rather than
	// linking the generated `CNetObjHandler`. `SnapshotDeltaSixup` is required by
	// `CDemoPlayer`'s constructor but never actually used (`IsSixup()` is only ever true for a
	// `0.7`-netversion demo, out of scope — see the task spec's goal), so it gets no static
	// sizes of its own.
	CSnapshotDelta SnapshotDelta;
	CSnapshotDelta SnapshotDeltaSixup;
	SnapshotDelta.SetStaticsize(NETOBJTYPE_PLAYERINPUT, sizeof(CNetObj_PlayerInput));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_PROJECTILE, sizeof(CNetObj_Projectile));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_LASER, sizeof(CNetObj_Laser));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_PICKUP, sizeof(CNetObj_Pickup));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_FLAG, sizeof(CNetObj_Flag));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_GAMEINFO, sizeof(CNetObj_GameInfo));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_GAMEDATA, sizeof(CNetObj_GameData));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_CHARACTERCORE, sizeof(CNetObj_CharacterCore));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_CHARACTER, sizeof(CNetObj_Character));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_PLAYERINFO, sizeof(CNetObj_PlayerInfo));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_CLIENTINFO, sizeof(CNetObj_ClientInfo));
	SnapshotDelta.SetStaticsize(NETOBJTYPE_SPECTATORINFO, sizeof(CNetObj_SpectatorInfo));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_COMMON, sizeof(CNetEvent_Common));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_EXPLOSION, sizeof(CNetEvent_Explosion));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_SPAWN, sizeof(CNetEvent_Spawn));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_HAMMERHIT, sizeof(CNetEvent_HammerHit));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_DEATH, sizeof(CNetEvent_Death));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_SOUNDGLOBAL, sizeof(CNetEvent_SoundGlobal));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_SOUNDWORLD, sizeof(CNetEvent_SoundWorld));
	SnapshotDelta.SetStaticsize(NETEVENTTYPE_DAMAGEIND, sizeof(CNetEvent_DamageInd));

	CDemoPlayer DemoPlayer(&SnapshotDelta, &SnapshotDeltaSixup, /*UseVideo=*/false);

	if(DemoPlayer.Load(&Storage, nullptr, pInputPath, IStorage::TYPE_ALL_OR_ABSOLUTE) == -1)
		Fail(std::string("Load failed: ") + DemoPlayer.ErrorMessage());

	FILE *pOut = fopen(pOutputPath, "w");
	if(!pOut)
		Fail(std::string("cannot open output file: ") + pOutputPath);

	const CDemoPlayer::CPlaybackInfo *pInfo = DemoPlayer.Info();
	const CMapInfo *pMapInfo = DemoPlayer.GetMapInfo();

	fprintf(pOut, "{\"header\":{\"version\":%d,\"netversion\":", (int)pInfo->m_Header.m_Version);
	JsonString(pOut, pInfo->m_Header.m_aNetversion);
	fputs(",\"map_name\":", pOut);
	JsonString(pOut, pMapInfo->m_aName);
	fprintf(pOut, ",\"map_size\":%u,\"map_crc\":%u,\"type\":", pMapInfo->m_Size, pMapInfo->m_Crc);
	JsonString(pOut, pInfo->m_Header.m_aType);
	fputs(",\"sha256\":", pOut);
	if(pMapInfo->m_Sha256.has_value())
	{
		char aShaStr[SHA256_MAXSTRSIZE];
		sha256_str(pMapInfo->m_Sha256.value(), aShaStr, sizeof(aShaStr));
		JsonString(pOut, aShaStr);
	}
	else
	{
		fputs("null", pOut);
	}
	fprintf(pOut, ",\"first_tick\":%d,\"last_tick\":%d}}\n", pInfo->m_Info.m_FirstTick, pInfo->m_Info.m_LastTick);

	CJsonListener Listener;
	Listener.m_pPlayer = &DemoPlayer;
	Listener.m_pOut = pOut;
	DemoPlayer.SetListener(&Listener);

	// `CDemoEditor::Slice` (`demo.cpp:1491-1499`) is the only real caller of this sequence in the
	// codebase: `Play()` once, then `Update(false)` in a loop until paused. A single
	// `Update(false)` call already drains the whole file on its own (its internal tick loop,
	// `demo.cpp:1248-1260`, only ever breaks early on `RealTime && ...`, and `RealTime` is
	// `false` here) — the `while` around it is defensive, matching the reference exactly rather
	// than relying on that single-call behavior holding forever.
	DemoPlayer.Play();
	while(DemoPlayer.IsPlaying())
	{
		DemoPlayer.Update(false);
		if(pInfo->m_Info.m_Paused)
			break;
	}
	DemoPlayer.Stop(); // required: `~CDemoPlayer` asserts `m_File == nullptr` (`demo.cpp:526`).

	fclose(pOut);
	return 0;
}
