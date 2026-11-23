package com.example.rainsense

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import kotlin.math.PI
import kotlin.math.sin

/** The Kotlin extractor must match rainmic/features.py, or the shipped model silently misbehaves. */
class ParityTest {
    @Test
    fun featuresMatchPython() {
        // same LCG noise + tones as parity_signal() in tests/test_rainmic.py
        var s = 12345L
        val sr = FeatureExtractor.SR
        val y = DoubleArray(sr) { i ->
            s = (s * 1664525L + 1013904223L) and 0xFFFFFFFFL
            (s.toDouble() / 4294967296.0 - 0.5) * 0.2 + 0.3 * sin(2 * PI * 440 * i / sr) + 0.05 * sin(2 * PI * 3000 * i / sr)
        }
        // Gradle runs unit tests from the module dir (android/rainsense)
        val expected = File("../../tests/fixtures/parity_features.txt").readLines().filter { it.isNotBlank() }.map { it.toDouble() }
        val actual = FeatureExtractor.compute(y)
        assertEquals(expected.size, actual.size)
        expected.forEachIndexed { i, e -> assertEquals("feature $i", e, actual[i], 1e-6) }
    }
}
