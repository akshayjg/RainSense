package com.example.rainmic

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.IBinder
import androidx.core.app.ServiceCompat
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
        val classifier = RainClassifier(this)
        val log = File(filesDir, "rain_log.csv")
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        while (scope.isActive) {
            try {
                val pcm = recordClip()
                val (label, probs) = classifier.classify(pcm)
                val ts = fmt.format(Date())
                val line = "$ts,$label," + probs.entries.joinToString(",") { "%s=%.2f".format(it.key, it.value) }
                log.appendText(line + "\n")
                RainState.text.value = "$ts\n$label\n" + probs.entries.joinToString("\n") { "%s: %.0f%%".format(it.key, it.value * 100) }
            } catch (e: Exception) {
                RainState.text.value = "Error: ${e.message}"
            }
            delay(INTERVAL_MS)
        }
    }

    @Suppress("MissingPermission") // permission is requested in MainActivity before starting the service
    private fun recordClip(): ShortArray {
        val total = FeatureExtractor.SR * FeatureExtractor.CLIP_SECONDS
        val minBuf = AudioRecord.getMinBufferSize(FeatureExtractor.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            FeatureExtractor.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 8192)
        )
        val out = ShortArray(total)
        try {
            rec.startRecording()
            var read = 0
            while (read < total) {
                val r = rec.read(out, read, total - read)
                if (r <= 0) throw IllegalStateException("AudioRecord read failed ($r)")
                read += r
            }
        } finally {
            rec.release()
        }
        return out
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
