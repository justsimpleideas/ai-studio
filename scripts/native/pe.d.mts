export interface PeExport {
  name: string
  data: boolean
}
export function readExports(buf: Buffer): PeExport[]
export function readImports(buf: Buffer): { normal: string[]; delayed: string[] }
export function moduleDefinition(dllName: string, exports: PeExport[]): string
