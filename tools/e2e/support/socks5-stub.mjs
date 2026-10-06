// A SOCKS5 stand-in for the local e2e (task 5.12): username/password login, and `UDP ASSOCIATE` answered with a relay on 127.0.0.1. It
// carries no traffic (the default relay mode of `proxy-check` opens no UDP socket), so «Проверить» can say ok or auth failed for real.
// Usage: node socks5-stub.mjs <port> <user> <pass>. Loopback only.
import net from "node:net";

const [, , portArg, user, pass] = process.argv;
const port = Number(portArg);
const server = net.createServer((sock) => {
  let stage = "greeting";
  let buf = Buffer.alloc(0);
  sock.on("error", () => {});
  sock.on("data", (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    for (;;) {
      if (stage === "greeting") {
        if (buf.length < 2 || buf.length < 2 + buf[1]) return;
        const methods = [...buf.subarray(2, 2 + buf[1])];
        buf = buf.subarray(2 + buf[1]);
        if (!methods.includes(2)) {
          sock.end(Buffer.from([5, 0xff]));
          return;
        }
        sock.write(Buffer.from([5, 2]));
        stage = "auth";
      } else if (stage === "auth") {
        if (buf.length < 2) return;
        const ulen = buf[1];
        if (buf.length < 2 + ulen + 1) return;
        const plen = buf[2 + ulen];
        if (buf.length < 3 + ulen + plen) return;
        const u = buf.subarray(2, 2 + ulen).toString();
        const p = buf.subarray(3 + ulen, 3 + ulen + plen).toString();
        buf = buf.subarray(3 + ulen + plen);
        if (u !== user || p !== pass) {
          sock.end(Buffer.from([1, 1]));
          return;
        }
        sock.write(Buffer.from([1, 0]));
        stage = "request";
      } else if (stage === "request") {
        if (buf.length < 10) return;
        // VER CMD RSV ATYP(1: IPv4) ADDR(4) PORT(2)
        const reply = Buffer.from([5, buf[1] === 3 ? 0 : 7, 0, 1, 127, 0, 0, 1, 0x1f, 0x90]);
        buf = Buffer.alloc(0);
        sock.write(reply);
        stage = "open";
      } else {
        return;
      }
    }
  });
});
server.listen(port, "127.0.0.1", () => console.log("socks5 stub on 127.0.0.1:" + port));
process.on("SIGTERM", () => server.close(() => process.exit(0)));
