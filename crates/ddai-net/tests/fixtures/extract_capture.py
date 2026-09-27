#!/usr/bin/env python3
"""Extract UDP payloads from a tcpdump pcap capture of DDNet loopback traffic into a compact
custom multi-segment fixture format (see crates/ddai-net/tests/capture.rs for the reader/format
docs).

A single contiguous packet range is not representative of a real DDNet session end to end (an
in-protocol map download in the middle can be thousands of packets on its own), so this tool
extracts one or more independently-contiguous *segments* — e.g. "the handshake" and, separately,
"a later steady-state window" — and keeps each segment's own packet ordering intact for
per-segment sequence/ack continuity checks.

Usage: extract_capture.py <input.pcap> <output.dat> <server_port> <start:end> [<start:end> ...]

`<start:end>` is an inclusive, 0-based packet-index range into the *filtered* stream of UDP
datagrams exchanged with `server_port` (i.e. index 0 is the first such datagram in the pcap, not
the first datagram overall) — matching what `tools/classify` (a throwaway diagnostic, not part of
this repo) would print for a raw capture.
"""
import struct
import sys


def read_pcap_records(path):
    with open(path, "rb") as f:
        global_header = f.read(24)
        magic = struct.unpack_from("<I", global_header, 0)[0]
        if magic == 0xa1b2c3d4:
            endian = "<"
        elif magic == 0xd4c3b2a1:
            endian = ">"
        else:
            raise ValueError(f"not a pcap file (magic={magic:#x})")
        linktype = struct.unpack_from(endian + "I", global_header, 20)[0]
        while True:
            rec_header = f.read(16)
            if len(rec_header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", rec_header)
            data = f.read(incl_len)
            yield linktype, ts_sec, ts_usec, data


def udp_payload(linktype, frame):
    # linktype 1 = DLT_EN10MB (Ethernet, possibly with zeroed MACs on loopback captures).
    if linktype != 1:
        raise ValueError(f"unsupported linktype {linktype}, expected DLT_EN10MB (1)")
    eth_len = 14
    if len(frame) < eth_len + 20 + 8:
        return None
    ethertype = struct.unpack_from(">H", frame, 12)[0]
    if ethertype != 0x0800:  # IPv4 only
        return None
    ip_start = eth_len
    ver_ihl = frame[ip_start]
    ihl = (ver_ihl & 0x0F) * 4
    proto = frame[ip_start + 9]
    if proto != 17:  # UDP
        return None
    udp_start = ip_start + ihl
    src_port, dst_port, udp_len, _csum = struct.unpack_from(">HHHH", frame, udp_start)
    payload_start = udp_start + 8
    payload_len = udp_len - 8
    payload = frame[payload_start : payload_start + payload_len]
    return src_port, dst_port, payload


def filtered_records(pcap_path, server_port):
    """All UDP datagrams exchanged with `server_port`, in capture order: (direction, payload)."""
    out = []
    for linktype, _ts_sec, _ts_usec, frame in read_pcap_records(pcap_path):
        parsed = udp_payload(linktype, frame)
        if parsed is None:
            continue
        src_port, dst_port, payload = parsed
        if server_port not in (src_port, dst_port):
            continue
        direction = 0 if dst_port == server_port else 1  # 0 = client->server, 1 = server->client
        out.append((direction, payload))
    return out


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    in_path, out_path, server_port_s = sys.argv[1:4]
    server_port = int(server_port_s)
    ranges = []
    for arg in sys.argv[4:]:
        start_s, end_s = arg.split(":")
        ranges.append((int(start_s), int(end_s)))

    all_records = filtered_records(in_path, server_port)
    segments = [all_records[start : end + 1] for start, end in ranges]

    with open(out_path, "wb") as f:
        f.write(b"DDCAP2")
        f.write(struct.pack("<B", len(segments)))
        for segment in segments:
            f.write(struct.pack("<I", len(segment)))
            for direction, payload in segment:
                f.write(struct.pack("<BH", direction, len(payload)))
                f.write(payload)

    total_records = sum(len(s) for s in segments)
    total_payload = sum(len(p) for s in segments for _, p in s)
    print(f"{len(segments)} segment(s), {total_records} records total, {total_payload} bytes of udp payload")
    for i, (segment, (start, end)) in enumerate(zip(segments, ranges)):
        print(f"  segment {i}: source records {start}..={end}, {len(segment)} records, {sum(len(p) for _, p in segment)} bytes")


if __name__ == "__main__":
    main()
