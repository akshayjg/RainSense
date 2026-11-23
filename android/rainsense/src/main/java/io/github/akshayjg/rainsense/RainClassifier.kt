package io.github.akshayjg.rainsense

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp

/** Loads assets/model.json (exported by rainsense.train) and runs softmax regression. */
class RainClassifier(context: Context) {
    private val classes: List<String>
    private val mean: DoubleArray
    private val scale: DoubleArray
    private val coef: Array<DoubleArray>
    private val intercept: DoubleArray
    private val rainThreshold: Double

    init {
        val text = context.assets.open("model.json").bufferedReader().use { it.readText() }
        val o = JSONObject(text)
        classes = (0 until o.getJSONArray("classes").length()).map { o.getJSONArray("classes").getString(it) }
        mean = o.getJSONArray("mean").toDoubles()
        scale = o.getJSONArray("scale").toDoubles()
        intercept = o.getJSONArray("intercept").toDoubles()
        val c = o.getJSONArray("coef")
        coef = Array(c.length()) { c.getJSONArray(it).toDoubles() }
        rainThreshold = o.optDouble("rain_threshold", 0.5)  // older models: plain argmax
    }

    /** Scores each 1 s window and takes the per-class median, so one loud second can't flip the result. Same as predict.predict_clip. */
    fun classify(pcm: ShortArray): Pair<String, Map<String, Double>> {
        val y = DoubleArray(pcm.size) { pcm[it] / 32768.0 }
        val w = FeatureExtractor.SR * FeatureExtractor.WINDOW_SECONDS
        val windows = if (y.size < w) listOf(y) else (0..y.size - w step w).map { y.copyOfRange(it, it + w) }
        val perWindow = windows.map { softmax(FeatureExtractor.compute(it)) }
        val med = DoubleArray(classes.size) { c -> median(perWindow.map { it[c] }) }
        val total = med.sum()
        val probs = classes.indices.associate { classes[it] to med[it] / total }
        return pickLabel(probs, rainThreshold) to probs
    }

    private fun softmax(f: DoubleArray): DoubleArray {
        val logits = DoubleArray(classes.size) { c ->
            var s = intercept[c]
            for (i in f.indices) s += coef[c][i] * (f[i] - mean[i]) / scale[i]
            s
        }
        val m = logits.max()
        val e = logits.map { exp(it - m) }
        val total = e.sum()
        return DoubleArray(classes.size) { e[it] / total }
    }

    private fun median(v: List<Double>): Double {
        val s = v.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }

    private fun JSONArray.toDoubles() = DoubleArray(length()) { getDouble(it) }
}

/** no_rain unless P(rain) = 1 - P(no_rain) reaches [threshold], then the likeliest rain class. Same rule as predict.label. */
internal fun pickLabel(probs: Map<String, Double>, threshold: Double): String {
    val noRain = probs["no_rain"] ?: return probs.maxByOrNull { it.value }!!.key
    if (1 - noRain < threshold) return "no_rain"
    return probs.filterKeys { it != "no_rain" }.maxByOrNull { it.value }!!.key
}
