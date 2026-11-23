package com.example.rainsense

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

/** Skips clips that are likely useless, before recording. A skipped clip beats a wrong one in the city map. */
internal object ContextGate {
    private const val VEHICLE_SPEED_MS = 7f        // ~25 km/h
    private const val FIX_MAX_AGE_MS = 120_000L
    private const val DARK_LUX = 5f

    suspend fun reasonToSkip(ctx: Context): String? = when {
        onWifi(ctx) -> "on Wi-Fi (probably indoors)"
        inVehicle(ctx) -> "moving fast (probably in a vehicle)"
        inPocket(ctx) -> "covered and dark (probably in a pocket or bag)"
        else -> null
    }

    private fun onWifi(ctx: Context): Boolean {
        val cm = ctx.getSystemService(ConnectivityManager::class.java)
        return cm.getNetworkCapabilities(cm.activeNetwork)?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
    }

    /** Uses the host app's location permission if it has one; otherwise this check is skipped. */
    @SuppressLint("MissingPermission")
    private fun inVehicle(ctx: Context): Boolean {
        val granted = listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            .any { ctx.checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED }
        if (!granted) return false
        val lm = ctx.getSystemService(LocationManager::class.java)
        val fix = lm.getProviders(true).mapNotNull { lm.getLastKnownLocation(it) }.maxByOrNull { it.time } ?: return false
        return System.currentTimeMillis() - fix.time < FIX_MAX_AGE_MS && fix.hasSpeed() && fix.speed > VEHICLE_SPEED_MS
    }

    private suspend fun inPocket(ctx: Context): Boolean {
        val sm = ctx.getSystemService(SensorManager::class.java)
        val prox = sm.getDefaultSensor(Sensor.TYPE_PROXIMITY) ?: return false
        val near = readOnce(sm, prox)?.let { it < prox.maximumRange } ?: return false
        val dark = sm.getDefaultSensor(Sensor.TYPE_LIGHT)?.let { s -> readOnce(sm, s)?.let { it < DARK_LUX } } ?: true
        return near && dark
    }

    private suspend fun readOnce(sm: SensorManager, sensor: Sensor): Float? = withTimeoutOrNull(500) {
        suspendCancellableCoroutine { cont ->
            val listener = object : SensorEventListener {
                override fun onSensorChanged(e: SensorEvent) {
                    sm.unregisterListener(this)
                    if (cont.isActive) cont.resume(e.values[0])
                }
                override fun onAccuracyChanged(s: Sensor, accuracy: Int) {}
            }
            sm.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL)
            cont.invokeOnCancellation { sm.unregisterListener(listener) }
        }
    }
}
