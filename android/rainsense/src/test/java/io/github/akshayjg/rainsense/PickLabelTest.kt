package io.github.akshayjg.rainsense

import org.junit.Assert.assertEquals
import org.junit.Test

/** Same cases as test_label_uses_rain_threshold in tests/test_rainsense.py. */
class PickLabelTest {
    private val probs = mapOf("no_rain" to 0.35, "light" to 0.40, "heavy" to 0.25)  // P(rain) = 0.65

    @Test fun rainWhenAboveThreshold() = assertEquals("light", pickLabel(probs, 0.6))
    @Test fun noRainWhenBelowThreshold() = assertEquals("no_rain", pickLabel(probs, 0.7))
}
