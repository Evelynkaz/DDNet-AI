import { readFileSync } from "node:fs";
import { inflate, inflateSync } from "node:zlib";

const MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024;

const HEADER_SIZE = 36;
const ITEM_TYPE_SIZE = 12;
const ITEM_HEADER_SIZE = 8;
const MAX_ITEM_TYPE = 0xffff;
const MAX_ITEM_ID = 0xffff;

export type DataFileItem = { type: number; id: number; data: DataView; sizeBytes: number };

export class DataFileReader {
  readonly version: number;
  readonly numItemTypes: number;
  readonly numItems: number;
  readonly numRawData: number;
  readonly itemSize: number;
  readonly dataSize: number;

  private readonly buf: Buffer;
  private readonly itemTypesOffset: number;
  private readonly itemOffsetsOffset: number;
  private readonly dataOffsetsOffset: number;
  private readonly dataSizesOffset: number;
  private readonly itemStart: number;
  private readonly dataStart: number;
  private readonly dataCache: Array<Buffer | undefined>;

  private constructor(buf: Buffer) {
    this.buf = buf;
    const fileSize = buf.length;
    if (fileSize < HEADER_SIZE) {
      throw new Error(`datafile: could not read file header. file truncated or not a datafile (size=${fileSize})`);
    }

    const magic = buf.toString("latin1", 0, 4);
    if (magic !== "DATA" && magic !== "ATAD") {
      throw new Error(`datafile: wrong header magic. magic=${JSON.stringify(magic)}`);
    }

    this.version = buf.readInt32LE(4);
    if (this.version !== 3 && this.version !== 4) {
      throw new Error(`datafile: unsupported header version. version=${this.version}`);
    }
    let headerSize = buf.readInt32LE(8);
    let swaplen = buf.readInt32LE(12);
    this.numItemTypes = buf.readInt32LE(16);
    this.numItems = buf.readInt32LE(20);
    this.numRawData = buf.readInt32LE(24);
    this.itemSize = buf.readInt32LE(28);
    this.dataSize = buf.readInt32LE(32);

    if (
      this.numItemTypes < 0 ||
      this.numItemTypes > MAX_ITEM_TYPE + 1 ||
      this.numItems < 0 ||
      this.numRawData < 0 ||
      this.itemSize < 0 ||
      this.itemSize % 4 !== 0 ||
      this.dataSize < 0
    ) {
      throw new Error(
        `datafile: invalid header information. num_types=${this.numItemTypes} num_items=${this.numItems} num_data=${this.numRawData} item_size=${this.itemSize} data_size=${this.dataSize}`,
      );
    }

    let size = 0;
    size += this.numItemTypes * ITEM_TYPE_SIZE;
    size += this.numItems * 4;
    size += this.numRawData * 4;
    let sizeFix = 0;
    if (this.version === 4) {

      sizeFix = this.numRawData * 4;
      size += sizeFix;
    }
    size += this.itemSize;

    if (HEADER_SIZE + size + this.dataSize !== fileSize) {
      throw new Error(`datafile: invalid header data size or truncated file. data_size=${this.dataSize} file_size=${fileSize}`);
    }

    const SIZE_OFFSET = 16;
    const headerFileSize = headerSize + SIZE_OFFSET;
    if (headerFileSize !== fileSize) {
      if (sizeFix !== 0 && headerFileSize + sizeFix === fileSize) {
        headerSize += sizeFix;
      } else {
        throw new Error(`datafile: invalid header size or truncated file. size=${headerFileSize} actual=${fileSize}`);
      }
    }

    const headerSwaplen = swaplen + SIZE_OFFSET;
    const fileSizeSwaplen = fileSize - this.dataSize;
    if (headerSwaplen !== fileSizeSwaplen) {
      if (swaplen % 4 === 0 && sizeFix !== 0 && headerSwaplen + sizeFix === fileSizeSwaplen) {
        swaplen += sizeFix;
      } else {
        throw new Error(`datafile: invalid header swaplen or truncated file. swaplen=${headerSwaplen} actual=${fileSizeSwaplen}`);
      }
    }

    this.itemTypesOffset = HEADER_SIZE;
    this.itemOffsetsOffset = this.itemTypesOffset + this.numItemTypes * ITEM_TYPE_SIZE;
    this.dataOffsetsOffset = this.itemOffsetsOffset + this.numItems * 4;
    if (this.version === 4) {
      this.dataSizesOffset = this.dataOffsetsOffset + this.numRawData * 4;
      this.itemStart = this.dataSizesOffset + this.numRawData * 4;
    } else {
      this.dataSizesOffset = -1;
      this.itemStart = this.dataOffsetsOffset + this.numRawData * 4;
    }
    this.dataStart = this.itemStart + this.itemSize;
    if (this.dataStart !== HEADER_SIZE + size) {
      throw new Error(`datafile: internal offset mismatch. data_start=${this.dataStart} expected=${HEADER_SIZE + size}`);
    }

    this.validate();
    this.dataCache = new Array(this.numRawData);
  }

  static open(path: string): DataFileReader {
    return new DataFileReader(readFileSync(path));
  }

  private validate(): void {
    let countedItems = 0;
    const usedTypes = new Set<number>();
    for (let i = 0; i < this.numItemTypes; i++) {
      const { type, start, num } = this.itemType(i);
      if (type < 0 || type > MAX_ITEM_TYPE) throw new Error(`datafile: item type has invalid type. index=${i} type=${type}`);
      if (usedTypes.has(type)) throw new Error(`datafile: item type has duplicate type. index=${i} type=${type}`);
      usedTypes.add(type);
      if (num <= 0) throw new Error(`datafile: item type has invalid number of items. index=${i} type=${type} num=${num}`);
      if (start !== countedItems) throw new Error(`datafile: item type has invalid start. index=${i} type=${type} start=${start}`);
      countedItems += num;
      if (countedItems > this.numItems) break;
    }
    if (countedItems !== this.numItems) {
      throw new Error(`datafile: mismatched number of items in item types. counted=${countedItems} header=${this.numItems}`);
    }

    let prev = -1;
    for (let i = 0; i < this.numItems; i++) {
      const off = this.itemOffset(i);
      if (i === 0 ? off !== 0 : off <= prev) throw new Error(`datafile: invalid item offset. index=${i} offset=${off} previous=${prev}`);
      if (off >= this.itemSize) throw new Error(`datafile: item offset larger than total item size. index=${i} offset=${off} total=${this.itemSize}`);
      prev = off;
    }

    let totalItemSize = 0;
    for (let t = 0; t < this.numItemTypes; t++) {
      const { type, start, num } = this.itemType(t);
      for (let i = start; i < start + num; i++) {
        const fileItemSize = this.fileItemSize(i);
        if (fileItemSize < ITEM_HEADER_SIZE) throw new Error(`datafile: map item too small for header. item_index=${i} size=${fileItemSize}`);
        const pos = this.itemStart + this.itemOffset(i);
        const typeAndId = this.buf.readUInt32LE(pos);
        const itemType = (typeAndId >>> 16) & MAX_ITEM_TYPE;
        const declared = this.buf.readInt32LE(pos + 4);
        if (itemType !== type) throw new Error(`datafile: mismatched item type. item_index=${i} type=${itemType} expected=${type}`);
        if (declared < 0 || declared % 4 !== 0 || declared !== fileItemSize - ITEM_HEADER_SIZE) {
          throw new Error(`datafile: map item size does not match file. item_index=${i} size=${declared} file_size=${fileItemSize - ITEM_HEADER_SIZE}`);
        }
        totalItemSize += fileItemSize;
        if (totalItemSize > this.itemSize) break;
      }
    }
    if (totalItemSize !== this.itemSize) {
      throw new Error(`datafile: mismatched total item size. expected=${totalItemSize} header=${this.itemSize}`);
    }

    prev = -1;
    for (let i = 0; i < this.numRawData; i++) {
      const off = this.dataOffset(i);
      if (i === 0 ? off !== 0 : off <= prev) throw new Error(`datafile: invalid data offset. index=${i} offset=${off} previous=${prev}`);
      if (off >= this.dataSize) throw new Error(`datafile: data offset larger than total data size. index=${i} offset=${off} total=${this.dataSize}`);
      prev = off;
    }
    if (this.dataSizesOffset >= 0) {
      for (let i = 0; i < this.numRawData; i++) {
        const s = this.buf.readInt32LE(this.dataSizesOffset + i * 4);
        if (s < 0) throw new Error(`datafile: data size invalid. index=${i} size=${s}`);
      }
    }
  }

  private itemType(i: number): { type: number; start: number; num: number } {
    const pos = this.itemTypesOffset + i * ITEM_TYPE_SIZE;
    return { type: this.buf.readInt32LE(pos), start: this.buf.readInt32LE(pos + 4), num: this.buf.readInt32LE(pos + 8) };
  }

  private itemOffset(i: number): number {
    return this.buf.readInt32LE(this.itemOffsetsOffset + i * 4);
  }

  private dataOffset(i: number): number {
    return this.buf.readInt32LE(this.dataOffsetsOffset + i * 4);
  }

  private fileItemSize(i: number): number {
    if (i === this.numItems - 1) return this.itemSize - this.itemOffset(i);
    return this.itemOffset(i + 1) - this.itemOffset(i);
  }

  private fileDataSize(i: number): number {
    if (i === this.numRawData - 1) return this.dataSize - this.dataOffset(i);
    return this.dataOffset(i + 1) - this.dataOffset(i);
  }

  getItemCount(): number {
    return this.numItems;
  }

  numData(): number {
    return this.numRawData;
  }

  getItem(index: number): DataFileItem {
    if (index < 0 || index >= this.numItems) {
      throw new RangeError(`datafile: invalid item index ${index} (num_items=${this.numItems})`);
    }
    const pos = this.itemStart + this.itemOffset(index);
    const typeAndId = this.buf.readUInt32LE(pos);
    const sizeBytes = this.fileItemSize(index) - ITEM_HEADER_SIZE;
    const payload = pos + ITEM_HEADER_SIZE;
    return {
      type: (typeAndId >>> 16) & MAX_ITEM_TYPE,
      id: typeAndId & MAX_ITEM_ID,
      data: new DataView(this.buf.buffer, this.buf.byteOffset + payload, sizeBytes),
      sizeBytes,
    };
  }

  findItems(type: number): Array<{ id: number; data: DataView; sizeBytes: number }> {
    for (let t = 0; t < this.numItemTypes; t++) {
      const it = this.itemType(t);
      if (it.type !== type) continue;
      const out: Array<{ id: number; data: DataView; sizeBytes: number }> = [];
      for (let i = it.start; i < it.start + it.num; i++) {
        const { id, data, sizeBytes } = this.getItem(i);
        out.push({ id, data, sizeBytes });
      }
      return out;
    }
    return [];
  }

  getData(index: number): Buffer {
    const cached = this.dataCache[index];
    if (cached !== undefined) return cached;
    const { raw, size } = this.rawData(index);
    let out: Buffer;
    if (size >= 0) {

      out = inflateSync(raw, { maxOutputLength: size });
      this.checkInflated(index, size, out);
    } else {
      out = Buffer.from(raw);
    }
    this.dataCache[index] = out;
    return out;
  }

  async getDataAsync(index: number): Promise<Buffer> {
    const cached = this.dataCache[index];
    if (cached !== undefined) return cached;
    const { raw, size } = this.rawData(index);
    let out: Buffer;
    if (size >= 0) {
      out = await new Promise<Buffer>((resolve, reject) => {
        inflate(raw, { maxOutputLength: size }, (err, res) => (err ? reject(err) : resolve(res)));
      });
      this.checkInflated(index, size, out);
    } else {
      out = Buffer.from(raw);
    }
    this.dataCache[index] = out;
    return out;
  }

  private rawData(index: number): { raw: Buffer; size: number } {
    if (index < 0 || index >= this.numRawData) {
      throw new RangeError(`datafile: invalid data index ${index} (num_data=${this.numRawData})`);
    }
    const start = this.dataStart + this.dataOffset(index);
    const raw = this.buf.subarray(start, start + this.fileDataSize(index));
    if (this.dataSizesOffset < 0) return { raw, size: -1 };

    const size = this.buf.readInt32LE(this.dataSizesOffset + index * 4);
    if (size <= 0 || size > MAX_UNCOMPRESSED_BYTES) {
      throw new Error(`datafile: data size invalid. index=${index} size=${raw.length} uncompressed=${size}`);
    }
    return { raw, size };
  }

  private checkInflated(index: number, size: number, out: Buffer): void {
    if (out.length !== size) {
      throw new Error(`datafile: failed to uncompress data. index=${index} wanted=${size} got=${out.length}`);
    }
  }
}
