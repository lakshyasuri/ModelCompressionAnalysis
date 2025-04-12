plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.llm.cnnimageclassifier"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.llm.cnnimageclassifier"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    implementation("org.pytorch:pytorch_android:1.13.0")
    implementation("org.pytorch:pytorch_android_torchvision:1.13.0") // For TensorImageUtils
    implementation("androidx.camera:camera-camera2:1.1.0")
    implementation("androidx.camera:camera-core:1.1.0")
    implementation("androidx.camera:camera-lifecycle:1.1.0")
    implementation("androidx.camera:camera-view:1.0.0-alpha29")  // Add this dependency
    implementation("androidx.exifinterface:exifinterface:1.3.6")
    implementation("com.github.bumptech.glide:glide:4.16.0")


}