package io.github.akshayjg.rainsense

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Records [samples] of mono 16-bit PCM at 16 kHz. Caller must hold RECORD_AUDIO. */
@Suppress("MissingPermission")
fun recordPcm(context: Context, samples: Int): ShortArray {
    // MIC often has AGC / noise suppression, which flattens loudness and removes rain hiss.
    val am = context.getSystemService(AudioManager::class.java)
    val source = if (am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true")
        MediaRecorder.AudioSource.UNPROCESSED else MediaRecorder.AudioSource.VOICE_RECOGNITION
    val sr = FeatureExtractor.SR
    val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    val rec = AudioRecord(source, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 8192))
    val out = ShortArray(samples)
    try {
        rec.startRecording()
        var read = 0
        while (read < samples) {
            val r = rec.read(out, read, samples - read)
            if (r <= 0) throw IllegalStateException("AudioRecord read failed ($r)")
            read += r
        }
    } finally {
        rec.release()
    }
    return out
}

/** Writes mono 16-bit PCM as a standard WAV file (readable by rainsense.data.load_mono_16k). */
fun writeWav(file: File, pcm: ShortArray, sr: Int = FeatureExtractor.SR) {
    val dataBytes = pcm.size * 2
    val buf = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
    buf.put("RIFF".toByteArray()).putInt(36 + dataBytes).put("WAVE".toByteArray())
    buf.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(sr).putInt(sr * 2).putShort(2).putShort(16)
    buf.put("data".toByteArray()).putInt(dataBytes)
    for (s in pcm) buf.putShort(s)
    file.parentFile?.mkdirs()
    file.writeBytes(buf.array())
}
