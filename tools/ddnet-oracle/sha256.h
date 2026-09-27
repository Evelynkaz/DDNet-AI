// SHA-256 (FIPS 180-4), a small self-contained implementation for Oracle A's own use: verifying
// a rawmap file's sha256 against what a scenario declares, and reporting a scenario file's own
// sha256 in the trace's metadata (see docs/formats.md). Not DDNet code — this is DDNet-AI's own
// original file (GPL-3.0-only, like the rest of this repository); SHA-256 is a public, unpatented
// algorithm with no license of its own to carry.
#ifndef DDNET_AI_ORACLE_SHA256_H
#define DDNET_AI_ORACLE_SHA256_H

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace oracle_sha256
{

inline uint32_t RotR(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// Processes exactly one 64-byte block, updating the 8-word state in place.
inline void ProcessBlock(uint32_t State[8], const unsigned char Block[64])
{
	static const uint32_t K[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
		0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
		0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
		0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
		0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
		0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
		0xc67178f2};

	uint32_t W[64];
	for(int i = 0; i < 16; i++)
	{
		W[i] = (uint32_t(Block[i * 4]) << 24) | (uint32_t(Block[i * 4 + 1]) << 16) | (uint32_t(Block[i * 4 + 2]) << 8) |
			uint32_t(Block[i * 4 + 3]);
	}
	for(int i = 16; i < 64; i++)
	{
		uint32_t S0 = RotR(W[i - 15], 7) ^ RotR(W[i - 15], 18) ^ (W[i - 15] >> 3);
		uint32_t S1 = RotR(W[i - 2], 17) ^ RotR(W[i - 2], 19) ^ (W[i - 2] >> 10);
		W[i] = W[i - 16] + S0 + W[i - 7] + S1;
	}

	uint32_t a = State[0], b = State[1], c = State[2], d = State[3];
	uint32_t e = State[4], f = State[5], g = State[6], h = State[7];
	for(int i = 0; i < 64; i++)
	{
		uint32_t S1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
		uint32_t Ch = (e & f) ^ (~e & g);
		uint32_t Temp1 = h + S1 + Ch + K[i] + W[i];
		uint32_t S0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
		uint32_t Maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t Temp2 = S0 + Maj;
		h = g;
		g = f;
		f = e;
		e = d + Temp1;
		d = c;
		c = b;
		b = a;
		a = Temp1 + Temp2;
	}
	State[0] += a;
	State[1] += b;
	State[2] += c;
	State[3] += d;
	State[4] += e;
	State[5] += f;
	State[6] += g;
	State[7] += h;
}

// Computes the SHA-256 digest of `Data` (`Len` bytes), as 32 raw bytes.
inline std::array<unsigned char, 32> Digest(const unsigned char *Data, size_t Len)
{
	uint32_t State[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

	size_t FullBlocks = Len / 64;
	for(size_t i = 0; i < FullBlocks; i++)
	{
		ProcessBlock(State, Data + i * 64);
	}

	// Final padded block(s): the remaining bytes, a 0x80 byte, zero padding, then the 64-bit
	// big-endian bit length — one block if that fits in 56 bytes, two if not.
	size_t Rem = Len - FullBlocks * 64;
	unsigned char Tail[128];
	std::memset(Tail, 0, sizeof(Tail));
	std::memcpy(Tail, Data + FullBlocks * 64, Rem);
	Tail[Rem] = 0x80;
	size_t TailBlocks = (Rem < 56) ? 1 : 2;
	uint64_t BitLen = uint64_t(Len) * 8;
	for(int i = 0; i < 8; i++)
	{
		Tail[TailBlocks * 64 - 1 - i] = (unsigned char)(BitLen >> (8 * i));
	}
	for(size_t i = 0; i < TailBlocks; i++)
	{
		ProcessBlock(State, Tail + i * 64);
	}

	std::array<unsigned char, 32> Out{};
	for(int i = 0; i < 8; i++)
	{
		Out[i * 4] = (unsigned char)(State[i] >> 24);
		Out[i * 4 + 1] = (unsigned char)(State[i] >> 16);
		Out[i * 4 + 2] = (unsigned char)(State[i] >> 8);
		Out[i * 4 + 3] = (unsigned char)(State[i]);
	}
	return Out;
}

inline std::array<unsigned char, 32> Digest(const std::vector<unsigned char> &Data)
{
	return Digest(Data.empty() ? nullptr : Data.data(), Data.size());
}

inline std::string ToHex(const std::array<unsigned char, 32> &Digest)
{
	static const char *Hex = "0123456789abcdef";
	std::string Out;
	Out.resize(64);
	for(int i = 0; i < 32; i++)
	{
		Out[i * 2] = Hex[Digest[i] >> 4];
		Out[i * 2 + 1] = Hex[Digest[i] & 0xf];
	}
	return Out;
}

} // namespace oracle_sha256

#endif
