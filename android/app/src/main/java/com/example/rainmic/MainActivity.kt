package com.example.rainmic

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.example.rainsense.FeatureExtractor
import com.example.rainsense.RainResult
import com.example.rainsense.Pitterpat
import com.example.rainsense.recordPcm
import com.example.rainsense.writeWav
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res[Manifest.permission.RECORD_AUDIO] == true) {
            ContextCompat.startForegroundService(this, Intent(this, RainService::class.java))
        } else {
            RainState.text.value = "Microphone permission is required"
        }
    }
    private val micPerm = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this).apply { textSize = 20f; gravity = Gravity.CENTER; setPadding(32, 64, 32, 64) }
        val start = Button(this).apply { text = "Start monitoring" }
        val stop = Button(this).apply { text = "Stop" }
        // what a host app would do: one call, with the context gate on
        val sampleNow = Button(this).apply { text = "Sample now (as a host app would)" }

        // Data collection: saves labelled clips to <external files>/recordings/<label>/<time>_<where>.wav,
        // the same layout rainmic.train and rainmic.compare read.
        val header = TextView(this).apply { text = "Collect training data: where is the phone?"; setPadding(0, 64, 0, 16) }
        val where = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, PLACES)
        }
        val labelButtons = LABELS.map { Button(this).apply { text = "Record ${it.replace('_', ' ')}" } }
        val collectStatus = TextView(this).apply { setPadding(0, 16, 0, 0) }

        setContentView(ScrollView(this).apply {
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(32, 96, 32, 32)
                addView(status); addView(start); addView(stop); addView(sampleNow)
                addView(header); addView(where)
                labelButtons.forEach { addView(it) }
                addView(collectStatus)
            })
        })

        start.setOnClickListener {
            val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33) perms += Manifest.permission.POST_NOTIFICATIONS
            permLauncher.launch(perms.toTypedArray())
        }
        stop.setOnClickListener {
            startService(Intent(this, RainService::class.java).setAction(RainService.ACTION_STOP))
        }
        sampleNow.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                micPerm.launch(Manifest.permission.RECORD_AUDIO)
                return@setOnClickListener
            }
            stopService(Intent(this, RainService::class.java))
            sampleNow.isEnabled = false
            RainState.text.value = "Listening for ${FeatureExtractor.CLIP_SECONDS} s..."
            lifecycleScope.launch {
                RainState.text.value = when (val r = Pitterpat.sample(this@MainActivity)) {
                    is RainResult.Reading -> "Rain probability ${"%.0f".format(r.rainProb * 100)}%\n(${r.label})"
                    is RainResult.Skipped -> "Skipped: ${r.reason}"
                }
                sampleNow.isEnabled = true
            }
        }
        labelButtons.zip(LABELS).forEach { (button, label) ->
            button.setOnClickListener {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    micPerm.launch(Manifest.permission.RECORD_AUDIO)
                    return@setOnClickListener
                }
                stopService(Intent(this, RainService::class.java)) // two recordings at once would leave one silent
                val place = where.selectedItem as String
                labelButtons.forEach { it.isEnabled = false }
                collectStatus.text = "Recording $label ($place) for ${COLLECT_SECONDS}s..."
                lifecycleScope.launch {
                    collectStatus.text = try {
                        val name = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date()) + "_$place.wav"
                        withContext(Dispatchers.IO) {
                            val pcm = recordPcm(this@MainActivity, FeatureExtractor.SR * COLLECT_SECONDS)
                            writeWav(File(getExternalFilesDir("recordings"), "$label/$name"), pcm)
                        }
                        "Saved $label/$name"
                    } catch (e: Exception) {
                        "Recording failed: ${e.message}"
                    }
                    labelButtons.forEach { it.isEnabled = true }
                }
            }
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) { RainState.text.collect { status.text = it } }
        }
    }

    companion object {
        private const val COLLECT_SECONDS = 15 // = 3 training clips of 5 s
        private val LABELS = listOf("no_rain", "light", "moderate", "heavy")
        // No underscores: rainmic.compare reads the place from the text after the last "_" in the filename.
        private val PLACES = listOf("hand", "pocket", "bag", "umbrella", "table", "other")
    }
}
