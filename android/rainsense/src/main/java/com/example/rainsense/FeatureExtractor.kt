package com.example.rainsense

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

/** Kotlin port of rainsense/features.py. Keep the two in sync (same constants, same order). */
object FeatureExtractor {
    const val SR = 16000
    const val N = 1024
    const val HOP = 512
    const val CLIP_SECONDS = 5
    const val WINDOW_SECONDS = 1

    private val BAND_BINS = arrayOf(
        6 to 8, 8 to 11, 11 to 15, 15 to 19, 19 to 25, 25 to 33, 33 to 44, 44 to 57,
        57 to 75, 75 to 99, 99 to 130, 130 to 171, 171 to 225, 225 to 296, 296 to 389, 389 to 512
    )
    private const val FLAT_LO = 64
    private const val FLAT_HI = 513
    private const val HF_LO = 128
    private const val HF_HI = 513

    private val window = DoubleArray(N) { 0.5 - 0.5 * cos(2 * PI * it / N) }

    /** y: mono 16 kHz samples scaled to [-1, 1]. Returns 19 features. */
    fun compute(input: DoubleArray): DoubleArray {
        val y = if (input.size < N) input.copyOf(N) else input
        val frames = (y.size - N) / HOP + 1
        val bandAcc = DoubleArray(BAND_BINS.size)
        var flatAcc = 0.0
        val hfDb = DoubleArray(frames)
        val re = DoubleArray(N)
        val im = DoubleArray(N)
        val spec = DoubleArray(N / 2 + 1)

        for (t in 0 until frames) {
            val off = t * HOP
            for (i in 0 until N) { re[i] = y[off + i] * window[i]; im[i] = 0.0 }
            fft(re, im)
            for (k in 0..N / 2) spec[k] = re[k] * re[k] + im[k] * im[k]

            for ((b, r) in BAND_BINS.withIndex()) {
                var s = 0.0
                for (k in r.first until r.second) s += spec[k]
                bandAcc[b] += s / (r.second - r.first)
            }
            var logSum = 0.0
            var sum = 0.0
            for (k in FLAT_LO until FLAT_HI) { val v = spec[k] + 1e-12; logSum += ln(v); sum += v }
            val n = (FLAT_HI - FLAT_LO).toDouble()
            flatAcc += exp(logSum / n) / (sum / n)

            var hf = 0.0
            for (k in HF_LO until HF_HI) hf += spec[k]
            hfDb[t] = 10 * log10(hf + 1e-10)
        }

        var sq = 0.0
        for (v in y) sq += v * v
        val rmsDb = 20 * log10(sqrt(sq / y.size) + 1e-8)
        val mean = hfDb.average()
        val std = sqrt(hfDb.sumOf { (it - mean) * (it - mean) } / frames)

        val out = DoubleArray(BAND_BINS.size + 3)
        for (b in BAND_BINS.indices) out[b] = 10 * log10(bandAcc[b] / frames + 1e-10)
        out[BAND_BINS.size] = flatAcc / frames
        out[BAND_BINS.size + 1] = rmsDb
        out[BAND_BINS.size + 2] = std
        return out
    }

    /** In-place iterative radix-2 FFT (size must be a power of two). */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2 * PI / len
            val wr = cos(ang); val wi = sin(ang)
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until len / 2) {
                    val a = i + k; val b = i + k + len / 2
                    val xr = re[b] * cr - im[b] * ci
                    val xi = re[b] * ci + im[b] * cr
                    re[b] = re[a] - xr; im[b] = im[a] - xi
                    re[a] += xr; im[a] += xi
                    val ncr = cr * wr - ci * wi
                    ci = cr * wi + ci * wr
                    cr = ncr
                }
                i += len
            }
            len = len shl 1
        }
    }
}
