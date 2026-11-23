package io.github.akshayjg.rainsense

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

sealed class RainResult {
    /** rainProb = 1 - P(no_rain), calibrated at training time. No audio is kept. */
    data class Reading(val rainProb: Double, val label: String, val probs: Map<String, Double>, val timeMs: Long) : RainResult()

    /** Nothing was classified, and why (no permission, in a pocket, in a vehicle, mic busy...). */
    data class Skipped(val reason: String) : RainResult()
}

/**
 * The whole public API. Call from the host app while it is in the foreground (Android blocks background mic use):
 *
 *     when (val r = RainSense.sample(context)) {
 *         is RainResult.Reading -> report(r.rainProb)
 *         is RainResult.Skipped -> log(r.reason)
 *     }
 */
object RainSense {
    @Volatile private var classifier: RainClassifier? = null

    /** Listens for 5 s and classifies on the phone. [useGate] = false skips the context checks (for testing). */
    suspend fun sample(context: Context, useGate: Boolean = true): RainResult = withContext(Dispatchers.IO) {
        val ctx = context.applicationContext
        if (ctx.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return@withContext RainResult.Skipped("no microphone permission")
        }
        if (useGate) ContextGate.reasonToSkip(ctx)?.let { return@withContext RainResult.Skipped(it) }

        val pcm = recordPcm(ctx, FeatureExtractor.SR * FeatureExtractor.CLIP_SECONDS)
        if (pcm.all { it == 0.toShort() }) return@withContext RainResult.Skipped("microphone busy (silent clip)")

        val c = classifier ?: RainClassifier(ctx).also { classifier = it }
        val (label, probs) = c.classify(pcm)
        val noRain = probs["no_rain"] ?: error("model.json has no no_rain class")
        RainResult.Reading(1.0 - noRain, label, probs, System.currentTimeMillis())
    }
}
