package com.example.rainmic

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.ServiceCompat
import com.example.rainsense.RainResult
import com.example.rainsense.Pitterpat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Latest result, observed by MainActivity. */
object RainState {
    val text = MutableStateFlow("Not running")
}

/** Foreground service: every INTERVAL_MS records a 5 s clip, classifies it, appends to rain_log.csv. */
class RainService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        startForegroundNotification()
        if (job == null) job = scope.launch { loop() }
        return START_STICKY
    }

    private fun startForegroundNotification() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Rain monitoring", NotificationManager.IMPORTANCE_LOW))
        val n = Notification.Builder(this, CHANNEL)
            .setContentTitle("RainMic is listening")
            .setContentText("Sampling the microphone every minute")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .build()
        ServiceCompat.startForeground(this, 1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
    }

    private suspend fun loop() {
        val log = File(filesDir, "rain_log.csv")
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        while (scope.isActive) {
            try {
                val ts = fmt.format(Date())
                // gate off: this harness is for a phone left in one place (often on Wi-Fi)
                when (val r = Pitterpat.sample(this@RainService, useGate = false)) {
                    is RainResult.Reading -> {
                        log.appendText("$ts,${r.label}," + r.probs.entries.joinToString(",") { String.format(Locale.US, "%s=%.2f", it.key, it.value) } + "\n")
                        RainState.text.value = "$ts\n${r.label}\n" + r.probs.entries.joinToString("\n") { "%s: %.0f%%".format(it.key, it.value * 100) }
                    }
                    is RainResult.Skipped -> {
                        log.appendText("$ts,skipped,${r.reason}\n")
                        RainState.text.value = "$ts\nskipped: ${r.reason}"
                    }
                }
            } catch (e: Exception) {
                RainState.text.value = "Error: ${e.message}"
            }
            delay(INTERVAL_MS)
        }
    }

    override fun onDestroy() {
        scope.cancel()
        RainState.text.value = "Stopped"
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "stop"
        private const val CHANNEL = "rain"
        private const val INTERVAL_MS = 60_000L
    }
}
