export type ParamsJSON = number[] | { dtype: "float32"; encoding: "base64"; length: number; data: string };

export function encodeParams(params: Float64Array): ParamsJSON {
  const f32 = Float32Array.from(params);
  return {
    dtype: "float32",
    encoding: "base64",
    length: f32.length,
    data: Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString("base64"),
  };
}

export function decodeParams(v: unknown): Float64Array {
  if (Array.isArray(v)) return Float64Array.from(v as number[]);
  const obj = v as { dtype?: unknown; encoding?: unknown; length?: unknown; data?: unknown } | null | undefined;
  if (obj?.dtype === "float32" && obj.encoding === "base64" && typeof obj.data === "string") {
    const bytes = Buffer.from(obj.data, "base64");

    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const f32 = new Float32Array(copy.buffer, 0, copy.byteLength >> 2);
    if (typeof obj.length === "number" && obj.length !== f32.length) {
      throw new Error(`decodeParams: header says ${obj.length} values, payload holds ${f32.length}`);
    }
    return Float64Array.from(f32);
  }
  throw new Error("decodeParams: unrecognised parameter encoding");
}
