package com.example.rainmic

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp

/** Loads assets/model.json (exported by rainmic.train) and runs softmax regression. */
class RainClassifier(context: Context) {
    private val classes: List<String>
    private val mean: DoubleArray
    private val scale: DoubleArray
    private val coef: Array<DoubleArray>
    private val intercept: DoubleArray

    init {
        val text = context.assets.open("model.json").bufferedReader().use { it.readText() }
        val o = JSONObject(text)
        classes = (0 until o.getJSONArray("classes").length()).map { o.getJSONArray("classes").getString(it) }
        mean = o.getJSONArray("mean").toDoubles()
        scale = o.getJSONArray("scale").toDoubles()
        intercept = o.getJSONArray("intercept").toDoubles()
        val c = o.getJSONArray("coef")
        coef = Array(c.length()) { c.getJSONArray(it).toDoubles() }
    }

    fun classify(pcm: ShortArray): Pair<String, Map<String, Double>> {
        val y = DoubleArray(pcm.size) { pcm[it] / 32768.0 }
        val f = FeatureExtractor.compute(y)
        val logits = DoubleArray(classes.size) { c ->
            var s = intercept[c]
            for (i in f.indices) s += coef[c][i] * (f[i] - mean[i]) / scale[i]
            s
        }
        val m = logits.max()
        val e = logits.map { exp(it - m) }
        val total = e.sum()
        val probs = classes.indices.associate { classes[it] to e[it] / total }
        return probs.maxByOrNull { it.value }!!.key to probs
    }

    private fun JSONArray.toDoubles() = DoubleArray(length()) { getDouble(it) }
}
