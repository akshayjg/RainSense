package com.example.rainmic

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res[Manifest.permission.RECORD_AUDIO] == true) {
            ContextCompat.startForegroundService(this, Intent(this, RainService::class.java))
        } else {
            RainState.text.value = "Microphone permission is required"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this).apply { textSize = 20f; gravity = Gravity.CENTER; setPadding(32, 64, 32, 64) }
        val start = Button(this).apply { text = "Start monitoring" }
        val stop = Button(this).apply { text = "Stop" }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 96, 32, 32)
            addView(status); addView(start); addView(stop)
        })

        start.setOnClickListener {
            val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33) perms += Manifest.permission.POST_NOTIFICATIONS
            permLauncher.launch(perms.toTypedArray())
        }
        stop.setOnClickListener {
            startService(Intent(this, RainService::class.java).setAction(RainService.ACTION_STOP))
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) { RainState.text.collect { status.text = it } }
        }
    }
}
