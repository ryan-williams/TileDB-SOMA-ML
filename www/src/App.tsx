import { Tooltip } from "@mui/material"
import { A, batched, ceil, E, floor, interp, log, log2, max, range, round, scan, shuffle, sum, } from "@rdub/base"
import { ClassName } from "@rdub/base/classname"
import { flatten } from "lodash"
import { Dispatch, forwardRef, KeyboardEventHandler, ReactNode, type SetStateAction, SVGProps, useEffect, useMemo, useState } from "react"
import seedrandom from 'seedrandom'
import useSessionStorageState, { SessionStorageState } from "use-session-storage-state"

const getBarW = interp([ 40, .9 ], [ 100, .5 ])

const Bar = forwardRef<
  SVGRectElement,
  {
    i: number,
    n: number,
    x: number,
    w: number,
    h: number,
    full: boolean,
  } & SVGProps<SVGRectElement>
>(
  ({ i, n, x, w, h, full, ...props }, ref) => {
    const fill = `hsl(${315 * i / n}, 100%, 50%)`
    return full ? (
      <rect
        {...props}
        ref={ref}
        width={w}
        height={h}
        x={x}
        fill={fill}
      />
    ) : (
      <path
        {...props}
        ref={ref}
        d={`M${x},0 h${w} v${h} h${-w} Z`}
        stroke={fill}
        // fill={fill}
        // fill={"white"}
        fillOpacity={.5}
        strokeWidth={.15}
        // strokeDasharray={.1}
      />
    )
  }
)

function getBarWX({ nBars, nChunks, }: {
  nBars: number
  nChunks: number
}): {
  w: number
  x: (i: number, chunkIdx: number) => number
} {
  const w = getBarW(nBars)
  const nEmpties = nChunks - 1
  const N = nBars + nEmpties
  const nGaps = N - 1
  const totalBarW = w * N
  const gapW = (100 - totalBarW) / nGaps
  const x = (i: number, chunkIdx: number) => {
    const idx = i + chunkIdx
    return idx * (w + gapW)
  }
  return { w, x }
}

export type BarTooltip = (_: { i: number }) => ReactNode
export type GroupTooltipArg = { idx: number, group: number[] }
export type GroupTooltip = (_: GroupTooltipArg) => ReactNode

export type BarsProps = {
  groups: number[][]
  n?: number
  y?: number
  h: number
  full?: boolean
  barTooltip?: BarTooltip
  groupTooltip?: GroupTooltip
  groupLabel?: string
} & ClassName

export function Bars({ groups, n, y = 0, h, full = false, barTooltip, groupTooltip, groupLabel, className }: BarsProps) {
  const groupLens = groups.map(g => g.length)
  n = n ?? sum(groupLens)
  const startIdxs = scan(groupLens, (acc, x) => acc + x, 0)
  if (groupLabel) {
    groupTooltip = ({ idx, group }: GroupTooltipArg) => <span>{groupLabel} {idx}: {group.join(", ")}</span>
  }
  const bars = getBarWX({ nBars: n, nChunks: groups.length })
  return (
    <svg viewBox={`0 0 100 ${h}`}>
      <g className={className} transform={`translate(0, ${y ?? 0})`}>{
        groups.map((group, groupIdx) => {
          const groupX0 = bars.x(startIdxs[groupIdx], groupIdx)
          const groupX1 = bars.x(startIdxs[groupIdx + 1], groupIdx)
          const groupWidth = groupX1 - groupX0
          const barGap = groupWidth / group.length - bars.w
          return (
            <g className={"shuffleChunk"} key={groupIdx}>
              {
                group.map((i, idx) => {
                  const xIdx = startIdxs[groupIdx] + idx
                  const bar = <Bar key={i} i={i} n={n} x={bars.x(xIdx, groupIdx)} w={bars.w} h={h} full={full} />
                  if (barTooltip) {
                    return <Tooltip arrow key={i} title={barTooltip({ i })}>{bar}</Tooltip>
                  } else {
                    return bar
                  }
                })
              }
              {groupTooltip && (
                <Tooltip arrow title={groupTooltip({ idx: groupIdx, group, })}>
                  <rect
                    x={groupX0 - barGap / 2}
                    y={0}
                    width={groupWidth}
                    height={h}
                    fill="transparent"
                    className="cursor-pointer"
                  />
                </Tooltip>
              )}
            </g>
          )
        })
      }</g>
    </svg>
  )
}

function Number({ label, min, state: { val, set } }: { label: ReactNode, min?: number, state: State<number> }) {
  const [ str, setStr ] = useState(val.toString())
  const [ err, setErr ] = useState(false)
  useEffect(() => {
    const val = parseInt(str)
    const err = isNaN(val) || (min !== undefined && val < min)
    setErr(err)
    if (!err) {
      set(val)
    }
  }, [ str, setErr, set, ])
  const onKeyDown: KeyboardEventHandler<HTMLInputElement> = e => {
    if (e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault()
      const currentVal = parseInt(str)
      if (!isNaN(currentVal)) {
        const newVal = currentVal + (e.key === 'ArrowUp' ? 10 : -10)
        const finalVal = min !== undefined ? Math.max(min, newVal) : newVal
        setStr(finalVal.toString())
      }
    }
  }
  return <label className={"number"}>
    <span>{label}</span>
    <input
      type={"number"}
      className={err ? "err" : undefined}
      value={str}
      style={{ width: `${max(2.1, str.length)}rem` }}
      min={min}
      onChange={e => { setStr(e.target.value) }}
      onKeyDown={onKeyDown}
    />
  </label>
}

// 0.00046% error at cutoff 1e5 (cf. `console.log`s below)
const DefaultCutoff = 1e5
function log2Factorial(n: number, cutoff: number = DefaultCutoff) {
  if (n <= cutoff) {
    if (n < 2) return 0;
    let sum = 0
    for (let i = 2; i < n; i+= 1) {
      sum += log(i)
    }
    return sum / log2(E)
  } else {
    // Stirling's approximation for log(n!) ≈ n*log(n) - n
    // Converting to log base 2: (n*ln(n) - n) / ln(2)
    return (n * log(n) - n) / log2(E);
  }
}
// const actualCutoff = log2Factorial(defaultCutoff)
// const estimatedCutoff = log2Factorial(defaultCutoff, defaultCutoff - 1)
// console.log(actualCutoff, estimatedCutoff, `${((estimatedCutoff - actualCutoff) / actualCutoff * 100).toPrecision(3)}%`)

export type Set<T> = Dispatch<SetStateAction<T>>
export type State<T> = { val: T, set: Set<T> }
function stateObj<T>([ val, set ]: [ T, Set<T> ] | SessionStorageState<T>): State<T> {
  return { val, set }
}

export type Controls = {
  n: State<number>
  shuffleChunkSize: State<number>
  ioBatchSize: State<number>
  seed?: State<number>
  miniBatchSize?: State<number>
  regenNonce?: State<number>
}

export type BaseDefaults = {
  n: number
  shuffleChunkSize: number
  ioBatchSize: number
}

function useBaseParams(name: string, defs: BaseDefaults) {
  const n = useSessionStorageState(`${name}.n`, { defaultValue: defs.n })
  const shuffleChunkSize = useSessionStorageState(`${name}.shuffleChunkSize`, { defaultValue: defs.shuffleChunkSize })
  const ioBatchSize = useSessionStorageState(`${name}.ioBatchSize`, { defaultValue: defs.ioBatchSize })
  return {
    n: stateObj(n),
    shuffleChunkSize: stateObj(shuffleChunkSize),
    ioBatchSize: stateObj(ioBatchSize),
  }
}

export type VizDefaults = {
  n: number
  shuffleChunkSize: number
  ioBatchSize: number
  miniBatchSize: number
}
function useVizParams(name: string, defs: VizDefaults) {
  const n = useSessionStorageState(`${name}.n`, { defaultValue: defs.n })
  const seed = useSessionStorageState(`${name}.seed`, { defaultValue: 0 })
  const shuffleChunkSize = useSessionStorageState(`${name}.shuffleChunkSize`, { defaultValue: defs.shuffleChunkSize })
  const ioBatchSize = useSessionStorageState(`${name}.ioBatchSize`, { defaultValue: defs.ioBatchSize })
  const miniBatchSize = useSessionStorageState(`${name}.miniBatchSize`, { defaultValue: defs.miniBatchSize })
  const regenNonce = useState(0)
  return {
    n: stateObj(n),
    seed: stateObj(seed),
    shuffleChunkSize: stateObj(shuffleChunkSize),
    ioBatchSize: stateObj(ioBatchSize),
    miniBatchSize: stateObj(miniBatchSize),
    regenNonce: stateObj(regenNonce),
  }
}

function Controls(
  {
    n,
    seed,
    shuffleChunkSize,
    ioBatchSize,
    miniBatchSize,
    regenNonce,
  }: Controls
) {
  const numShuffleChunks = ceil(n.val / shuffleChunkSize.val)
  const numFullIOBatches = floor(n.val / ioBatchSize.val)
  let ioBatchBits = numFullIOBatches * log2Factorial(ioBatchSize.val)
  const extra = n.val % ioBatchSize.val
  if (extra) {
    ioBatchBits += log2Factorial(extra)
  }
  const idealBits = useMemo(() => log2Factorial(n.val), [n.val])
  const shuffleChunkBits = useMemo(() => log2Factorial(numShuffleChunks), [ numShuffleChunks ])
  const actualBits = useMemo(() => shuffleChunkBits + ioBatchBits, [ shuffleChunkBits, ioBatchBits ])
  function formatBits(bits: number) {
    return round(bits)
  }
  return <>
    <div className={"controls"}>
      <Number label={"N"} min={1} state={n} />
      {seed && <Number label={"Seed"} state={seed} />}
      <Number label={"Shuffle chunk"} min={1} state={shuffleChunkSize} />
      <Number label={"IO batch"} min={1} state={ioBatchSize} />
      {miniBatchSize && <Number label={"Mini-batch"} min={1} state={miniBatchSize} />}
      {regenNonce && <input type={"button"} value={"Shuffle"} onClick={() => regenNonce.set((nonce: number) => nonce + 1) } />}
    </div>
    <div>
      <p>Ideal shuffle entropy: {formatBits(idealBits).toLocaleString()} bits</p>
      <p>Actual: {formatBits(actualBits).toLocaleString()} ({round(100 * actualBits / idealBits)}%; {formatBits(shuffleChunkBits).toLocaleString()} from shuffling chunks, {formatBits(ioBatchBits).toLocaleString()} from shuffling IO batches)</p>
    </div>
  </>
}

const VizDefs: VizDefaults = { n: 100, shuffleChunkSize: 5, ioBatchSize: 20, miniBatchSize: 4, }
const BaseDefs: BaseDefaults = { n: 100000, shuffleChunkSize: 64, ioBatchSize: 65536, }

function App() {
  const viz = useVizParams("viz", VizDefs)
  const base = useBaseParams("defs", BaseDefs)
  const {
    n: { val: n, },
    seed: { val: seed, },
    shuffleChunkSize: { val: shuffleChunkSize, },
    ioBatchSize: { val: ioBatchSize, },
    miniBatchSize: { val: miniBatchSize, },
    regenNonce: { val: regenNonce, },
  } = viz
  const rng = useMemo(
    // Need a terminator after stringified number: https://github.com/davidbau/seedrandom/issues/48
    () => seedrandom(`${seed + regenNonce}\n`),
    [ seed, regenNonce ]
  )

  const barH = 3
  const idxs = useMemo(() => range(n), [n])
  const shuffleChunks = useMemo(() => shuffle(batched(idxs, shuffleChunkSize), rng), [ idxs, shuffleChunkSize, rng ])
  const ioBatches = useMemo(() => batched(flatten(shuffleChunks), ioBatchSize).map(ioBatch => shuffle(ioBatch, rng)), [shuffleChunks, ioBatchSize, rng])
  const miniBatches = useMemo(() => batched(flatten(ioBatches), miniBatchSize), [ioBatches, miniBatchSize])
  const Row = (props: Omit<BarsProps, 'h'>) => <Bars {...props} h={barH} />
  const IOBatchIterable = <A href={"https://single-cell-data.github.io/TileDB-SOMA-ML/#module-tiledbsoma_ml._io_batch_iterable"}><code>IOBatchIterable</code></A>
  return (
    <>
      <div className="container">
        <h1><A href={"https://github.com/single-cell-data/TileDB-SOMA-ML"}>TileDB-SOMA-ML</A> shuffle simulator</h1>
        <p>Visualization / Analysis of a chunked shuffle used to stream remote, out-of-core <A href={"https://github.com/single-cell-data/TileDB-SOMA"}>TileDB-SOMA</A> sparse matrices into PyTorch.</p>
        <p>1. Given <code>int64</code> indices of cells in a TileDB-SOMA experiment (e.g. returned from an <A href={"https://tiledbsoma.readthedocs.io/en/1.15.0/python-tiledbsoma-experimentaxisquery.html"}><code>ExperimentAxisQuery</code></A>:</p>
        <Row
          groups={[idxs]}
          barTooltip={({ i }) => <span>Row {i}</span>}
        />
        <p>2. Break the cell indices into "shuffle chunks", and shuffle those (<A href={"https://single-cell-data.github.io/TileDB-SOMA-ML/#module-tiledbsoma_ml._query_ids"}><code>QueryIDs</code></A>):</p>
        <Row
          groups={shuffleChunks}
          groupTooltip={
            ({ idx, group }) =>
              <span>Shuffle chunk {idx}: [{group[0]}, {group[group.length - 1] + 1}</span>
          }
        />
        <p>3. Group those into "IO batches", and shuffle within each ({IOBatchIterable}):</p>
        <Row groups={ioBatches} groupLabel={"IO batch"} />
        <p>4. Iterate over IO batches, fetching <code>obs</code> and <code>X</code> data for each cell ({IOBatchIterable}):</p>
        <Row groups={ioBatches} groupLabel={"IO batch"} full />
        <p>5. Stride through each "IO batch", emitting "mini-batches" to the GPU (<A href={"https://single-cell-data.github.io/TileDB-SOMA-ML/#module-tiledbsoma_ml._mini_batch_iterable"}><code>MiniBatchIterable</code></A>):</p>
        <Row groups={miniBatches} groupLabel={"Mini-batch"} full />
        <Controls {...viz} />
        <hr/>
        <h2>Production-sized example</h2>
        <p>
          <A href={"https://github.com/single-cell-data/TileDB-SOMA-ML/blob/v0.1.0/src/tiledbsoma_ml/dataset.py#L31-L32"}>The defaults</A> in TileDB-SOMA-ML are 2^6 (shuffle chunk size) and 2^16 (IO batch size).
          <br/>
          This provides most of the entropy of a full shuffle, for 1e5-1e8 sized queries:
        </p>
        <Controls {...base} />
      </div>
    </>
  )
}

export default App
