package com.example.currency_classifier

import android.Manifest
import android.app.Activity
import android.app.Activity.*
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import com.example.currency_classifier.ml.Model
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.appcompat.app.AppCompatActivity.RESULT_OK as RESULT_OK1
import androidx.appcompat.app.AppCompatActivity.RESULT_OK as RESULT_OK1
import androidx.appcompat.app.AppCompatActivity.RESULT_OK as RESULT_OK1
import androidx.core.app.ActivityCompat

import android.content.pm.PackageManager

import androidx.core.content.ContextCompat




private const val FILE_NAME = "photo.jpg"
private const val REQUEST_CODE = 3
private lateinit var photoFile: File
private lateinit var takenImage:Bitmap
private val dict = mapOf(
    0 to "2 hundred",
    1 to "hundred",
    2 to "2 thousand",
    3 to "fifty",
    4 to "twenty",
    5 to "ten",
    6 to "5 hundred"
)

class MainActivity : AppCompatActivity() {
    private var mUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if(!checkPermission()){
            requestPermission()
        }


    }
    private fun checkPermission(): Boolean {
        return if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            // Permission is not granted
            false
        } else true
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA),
            REQUEST_CODE
        )
    }

    fun getImage(view: android.view.View) {

            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            photoFile = getPhotoFile(FILE_NAME)
//        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoFile)
            val fileProvider=FileProvider.getUriForFile(this,"com.example.currency_classifier.fileprovider",
                photoFile)
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            if(takePictureIntent.resolveActivity(this.packageManager)!=null) {
                startActivityForResult(takePictureIntent, REQUEST_CODE)
            }
            else{
                Toast.makeText(this,"Unable to open camera",Toast.LENGTH_SHORT).show()
            }



    }

    private fun getPhotoFile(fileName: String): File {
        val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(fileName,".jpg",storageDirectory)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if( requestCode == REQUEST_CODE && resultCode==Activity.RESULT_OK){
//            val takenImage = data?.extras?.get("data") as Bitmap
             takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)
             image_view.setImageBitmap(takenImage)
        }
        super.onActivityResult(requestCode, resultCode, data)
    }

    fun predict(view: View) {
        if(image_view.drawable==null){
            Toast.makeText(this,"Click a picture of an Indian Currency Note",Toast.LENGTH_SHORT).show()
        }
        else{
            val img = takenImage
            try {
                val model = Model.newInstance(this)


                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

                val byteBuffer = convertBitmapToByteBuffer(img)

                Log.d("shape", byteBuffer.toString())
                Log.d("shape", inputFeature0.buffer.toString())

                if (byteBuffer != null) {
                    inputFeature0.loadBuffer(byteBuffer)
                }


                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer


                model.close()
                val nums = outputFeature0.floatArray
                textView.text = dict[findInx(nums)]


                Log.d("Tag", dict[findInx(nums)].toString())
            } catch (e: IOException) {
                // TODO Handle the exception
            }
        }

    }
    private fun findInx(arr: FloatArray) :Int{
        var index = 0
        var max: Float
        max = arr[0]
        for(i in 0..arr.size-1){
            if(arr[i]>max){
                index=i
                max=arr[i]
            }
        }
        return index
    }



    private fun convertBitmapToByteBuffer(bp: Bitmap): ByteBuffer? {
        val imgData = ByteBuffer.allocateDirect(java.lang.Float.BYTES * 224 * 224 * 3)
        imgData.order(ByteOrder.nativeOrder())
        val bitmap = Bitmap.createScaledBitmap(bp, 224, 224, true)
        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Convert the image to floating point.
        var pixel = 0
        for (i in 0..223) {
            for (j in 0..223) {
                val `val` = intValues[pixel++]
                imgData.putFloat((`val` shr 16 and 0xFF) / 255f)
                imgData.putFloat((`val` shr 8 and 0xFF) / 255f)
                imgData.putFloat((`val` and 0xFF) / 255f)
            }
        }
        return imgData
    }



}