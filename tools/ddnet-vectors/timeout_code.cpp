// Portions copied from DDNet 20.1 (zlib licence, (c) the DDNet authors); altered: extracted into a stand-alone harness.
// Reference vectors for the bot's timeout code (task 4.10, D-100): DDNet 20.1's own algorithm, copied verbatim.
//   engine/client/client.cpp  GenerateTimeoutCode   (MD5 over "normal\0", seed + "\0", the raw NETADDR bytes)
//   base/secure.cpp           generate_password
//   base/types.h              NETADDR (24 bytes: type, ip[16], port, two bytes of padding; zeroed first, as CClient::Connect does)
//   engine/external/md5       DDNet's bundled MD5
// Build and run (needs the DDNet 20.1 source tree, which is not in this repo):
//   S=~/aiddnet/build/ddnet-20.1/src/src
//   gcc -c -I$S/engine/external/md5 $S/engine/external/md5/md5.c -o md5.o
//   g++ -I$S/engine/external/md5 tools/ddnet-vectors/timeout_code.cpp md5.o -o vec && ./vec
// The output lines are the `CPP_VECTORS` of `crates/ddai-net/src/timeout_code.rs`.
#include <cstdio>
#include <cstring>
#include <cstdint>
extern "C" {
#include "md5.h"
}
typedef struct NETADDR { unsigned int type; unsigned char ip[16]; unsigned short port; } NETADDR;
static void generate_password(char *buffer, unsigned length, const unsigned short *random, unsigned random_length)
{
	static const char VALUES[] = "ABCDEFGHKLMNPRSTUVWXYZabcdefghjkmnopqt23456789";
	static const size_t NUM_VALUES = sizeof(VALUES) - 1;
	buffer[random_length * 2] = 0;
	for(unsigned i = 0; i < random_length; i++)
	{
		unsigned short random_number = random[i] % 2048;
		buffer[2 * i + 0] = VALUES[random_number / NUM_VALUES];
		buffer[2 * i + 1] = VALUES[random_number % NUM_VALUES];
	}
}
static void code(const char *seed, const NETADDR *a, int n, bool dummy)
{
	md5_state_t st; md5_init(&st);
	const char *d = dummy ? "dummy" : "normal";
	md5_append(&st, (const md5_byte_t *)d, strlen(d) + 1);
	md5_append(&st, (const md5_byte_t *)seed, strlen(seed) + 1);
	for(int i = 0; i < n; i++) md5_append(&st, (const md5_byte_t *)&a[i], sizeof(a[i]));
	md5_byte_t dig[16]; md5_finish_(&st, dig);
	unsigned short r[8]; memcpy(r, dig, sizeof(r));
	char out[33]; generate_password(out, sizeof(out), r, 8);
	printf("%s %s %d ", seed, dummy ? "dummy" : "normal", (int)sizeof(NETADDR));
	for(int i = 0; i < 1; i++) printf("type=%u port=%u ip=%d.%d.%d.%d ", a[i].type, a[i].port, a[i].ip[0], a[i].ip[1], a[i].ip[2], a[i].ip[3]);
	printf("-> %s\n", out);
}
int main()
{
	NETADDR a; 
	memset(&a, 0, sizeof(a)); a.type = 1; a.ip[0]=127; a.ip[3]=1; a.port = 8443;
	code("ABCDEFGHKLMNPRST", &a, 1, false);
	code("ABCDEFGHKLMNPRST", &a, 1, true);
	memset(&a, 0, sizeof(a)); a.type = 1; a.ip[0]=192; a.ip[1]=0; a.ip[2]=2; a.ip[3]=35; a.port = 8308;
	code("ABCDEFGHKLMNPRST", &a, 1, false);
	code("n2e9mUWqk3HdGPt8", &a, 1, false);
	memset(&a, 0, sizeof(a)); a.type = 1; a.ip[0]=127; a.ip[3]=1; a.port = 8444;
	code("n2e9mUWqk3HdGPt8", &a, 1, false);
	NETADDR b; memset(&b, 0, sizeof(b)); b.type = 2; for(int i=0;i<16;i++) b.ip[i]=(unsigned char)(i==15?1:0); b.port = 8303;
	code("n2e9mUWqk3HdGPt8", &b, 1, false);
	return 0;
}
