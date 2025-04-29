import { Tooltip } from "@mui/material"
import { A, batched, interp, log, range, round, scan, shuffle, sum, } from "@rdub/base"
import { ClassName } from "@rdub/base/classname"
import { flatten } from "lodash"
import { Dispatch, forwardRef, KeyboardEventHandler, ReactNode, SVGProps, useEffect, useMemo, useState } from "react"
import seedrandom from 'seedrandom'
import useSessionStorageState from "use-session-storage-state"

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

function Number({ label, min, state: [ val, set ] }: { label: ReactNode, min?: number, state: [ number, Dispatch<number> ] }) {
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
      width={5}
      min={min}
      onChange={e => { setStr(e.target.value) }}
      onKeyDown={onKeyDown}
    />
  </label>
}

function log2Factorial(n: number) {
  return sum(range(2, n).map(i => log(i))) / log(2)
}

function App() {
  const [ n, setN ] = useSessionStorageState("n", { defaultValue: 100 })
  const [ seed, setSeed ] = useSessionStorageState("seed", { defaultValue: 0 })
  const [ shuffleChunkSize, setShuffleChunkSize ] = useSessionStorageState("shuffleChunkSize", { defaultValue: 5 })
  const [ ioBatchSize, setIoBatchSize ] = useSessionStorageState("ioBatchSize", { defaultValue: 20 })
  const [ miniBatchSize, setMiniBatchSize ] = useSessionStorageState("miniBatchSize", { defaultValue: 4 })
  const [ regenNonce, setRegenNonce ] = useState(0)
  const rng = useMemo(
    // Need a terminator after stringified number: https://github.com/davidbau/seedrandom/issues/48
    () => seedrandom(`${seed + regenNonce}\n`),
    [ seed, regenNonce ]
  )

  const barH = 3
  const barsGap = 4
  const idxs = useMemo(() => range(n), [n])
  const shuffleChunks = useMemo(() => shuffle(batched(idxs, shuffleChunkSize), rng), [ idxs, shuffleChunkSize, rng ])
  const ioBatches = useMemo(() => batched(flatten(shuffleChunks), ioBatchSize).map(ioBatch => shuffle(ioBatch, rng)), [shuffleChunks, ioBatchSize, rng])
  const miniBatches = useMemo(() => batched(flatten(ioBatches), miniBatchSize), [ioBatches, miniBatchSize])
  const Row = (props: Omit<BarsProps, 'h'>) => <Bars {...props} h={barH} />

  const idealBits = useMemo(() => log2Factorial(n), [n])
  const shuffleChunkBits = useMemo(() => log2Factorial(shuffleChunks.length), [ shuffleChunks.length ])
  const ioBatchBits = useMemo(() => sum(ioBatches.map(ioBatch => log2Factorial(ioBatch.length))), [ ioBatches ])
  const actualBits = useMemo(() => shuffleChunkBits + ioBatchBits, [ shuffleChunkBits, ioBatchBits ])
  function formatBits(bits: number) {
    return round(bits)
  }

  return (
    <>
      <div className="container">
        <div>
          <h1><A href={"https://github.com/single-cell-data/TileDB-SOMA-ML"}>TileDB-SOMA-ML</A> shuffle simulator</h1>
        </div>
        <p>Given indices of cells in a TileDB-SOMA experiment<br/>(each corresponding to a cell, with metadata at a given row in the <code>obs</code> DataFrame, and gene expression data in the corresponding row of the <code>X</code> matrix):</p>
        <Row
          groups={[idxs]}
          barTooltip={({ i }) => <span>Row {i}</span>}
        />
        <p>Break the cell indices into "shuffle chunks", and shuffle those:</p>
        <Row
          groups={shuffleChunks}
          groupTooltip={
            ({ idx, group }) =>
              <span>Shuffle chunk {idx}: [{group[0]}, {group[group.length - 1] + 1}</span>
          }
        />
        <p>Group those into "IO batches", and shuffle within each:</p>
        <Row groups={ioBatches} groupLabel={"IO batch"} />
        <p>
          Now, fetch the <code>obs</code> and <code>X</code> data for each cell
          <br/>
          (in reality, this is done lazily, by an <code>Iterator</code>, with one "IO chunk" of pre-fetching)
        </p>
        <Row groups={ioBatches} groupLabel={"IO batch"} full />
        <p>Finally, stride through each "IO batch", emitting "mini-batches" to the GPU:</p>
        <Row groups={miniBatches} groupLabel={"Mini-batch"} full />
        <div className={"controls"}>
          <Number label={"N"} min={1} state={[ n, setN ]} />
          <Number label={"Seed"} state={[ seed, setSeed ]} />
          <Number label={"Shuffle chunk"} min={1} state={[ shuffleChunkSize, setShuffleChunkSize ]} />
          <Number label={"IO batch"} min={1} state={[ ioBatchSize, setIoBatchSize ]} />
          <Number label={"Mini-batch"} min={1} state={[ miniBatchSize, setMiniBatchSize ]} />
          <input type={"button"} value={"Shuffle"} onClick={() => setRegenNonce(nonce => nonce + 1)} />
        </div>
        <div>
          <p>Ideal shuffle entropy: {formatBits(idealBits)} bits</p>
          <p>Actual: {formatBits(actualBits)} ({round(100 * actualBits / idealBits)}%; {formatBits(shuffleChunkBits)} from shuffling chunks, {formatBits(ioBatchBits)} from shuffling IO batches)</p>
        </div>
      </div>
    </>
  )
}

export default App
