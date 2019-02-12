/*
 * Copyright 2018 Zihua Zeng (edvard_hua@live.com)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example;

import android.app.Activity;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.KalmanFilter;

import com.xiaomi.mace.JniMaceUtils;

import java.io.IOException;


public class PoseEstimationFloatInception extends PoseEstimation {
    private Mat mMat;

    const static final int jointNum = 14;      // Depends on the pose estimatation model outputs.
    private int KF_stateNum = 56;
    private int KF_measureNum = 28;
    private KalmanFilter KF = new KalmanFilter(KF_stateNum, KF_measureNum, CvType.CV_32F);

    /**
     * Initializes an {@code PoseEstimation}.
     *
     * @param activity
     */
    PoseEstimationFloatInception(Activity activity) throws IOException {
        super(activity);
        initKalmanFilter();
    }


    private void initKalmanFilter(){
        Mat tM = new Mat.eye(KF_stateNum, KF_stateNum, CvType.CV_32F); //Construct transitionMatrix
        for (int i = 0; i < KF_measureNum; i++){
            tM[i, i+KF_measureNum] = 1.0f;
        } 
        KF.set_transitionMatrix(tM);

        Mat mM = new Mat.eye(KF_measureNum, KF_stateNum, CvType.CV_32F); //Construct measurementMatrix
        KF.set_measurementMatrix(mM);

        Mat mStPost = new Mat(KF_stateNum, 1, CvType.CV_32F, new Scalar(1e-1)); //Construct State matrix
        KF.set_statePre(mStPost);
    }

    @Override
    protected int getImageSizeX() {
        return 192;
    }

    @Override
    protected int getImageSizeY() {
        return 192;
    }

    @Override
    protected int getOutputSizeX() {
        return 96;
    }

    @Override
    protected int getOutputSizeY() {
        return 96;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        //bgr
        floatBuffer.put(pixelValue & 0xFF);
        floatBuffer.put((pixelValue >> 8) & 0xFF);
        floatBuffer.put((pixelValue >> 16) & 0xFF);
    }


    @Override
    protected void runInference() {
        float[] result = JniMaceUtils.maceMobilenetClassify(floatBuffer.array());

        if (mPrintPointArray == null)
            mPrintPointArray = new float[2][jointNum];

        if (!CameraActivity.isOpenCVInit)
            return;

        //先进行高斯滤波,5*5
        if (mMat == null)
            mMat = new Mat(96, 96, CvType.CV_32F);

        float[] tempArray = new float[getOutputSizeY() * getOutputSizeX()];
        float[] outTempArray = new float[getOutputSizeY() * getOutputSizeX()];

        long st = System.currentTimeMillis();

        for (int i = 0; i < jointNum; i++) {
            int index = 0;
            for (int x = 0; x < 96; x++) {
                for (int y = 0; y < 96; y++) {
                    tempArray[index] = result[x * getOutputSizeY() * jointNum + y * jointNum + i];
                    index++;
                }
            }

            mMat.put(0, 0, tempArray);
            Imgproc.GaussianBlur(mMat, mMat, new Size(5, 5), 0, 0);
            mMat.get(0, 0, outTempArray);

            float maxX = 0, maxY = 0;
            float max = 0;

            for (int x = 0; x < getOutputSizeX(); x++) {
                for (int y = 0; y < getOutputSizeY(); y++) {
                    float center = get(x, y, outTempArray);

                    if (center >= 0.01) {

                        if (center > max) {
                            max = center;
                            maxX = x;
                            maxY = y;
                        }
                    }
                }
            }

            if (max == 0) {
                mPrintPointArray = new float[2][jointNum];
                return;
            }

            mPrintPointArray[0][i] = maxY;
            mPrintPointArray[1][i] = maxX;
        }

        
        // Kalman filter -- prediction
        Mat prediction = KF.predict()

        // Kalman filter -- update
        Mat tmpMat = flatten2d(mPrintPointArray);
        KF.correct(tmpMat);

        // ** update mPrintPointArray after KF.correct(), since mPrintPointArray will be used in KF.correct()
        for(int i = 0; i < jointNum; i++){
            mPrintPointArray[0][i] = prediction.get(2*i, 0)[0];
            mPrintPointArray[1][i] = prediction.get(2*i+1, 0)[0];
        }

        Log.i("post_processing", "" + (System.currentTimeMillis() - st));
    }

    private float get(int x, int y, float[] arr) {
        if (x < 0 || y < 0 || x >= getOutputSizeX() || y >= getOutputSizeY())
            return -1;
        return arr[x * getOutputSizeX() + y];
    }

    private Mat flatten2d(float[][] arr){
        int rows = arr.length;
        int cols = arr[0].length;
        Mat result = new Mat(rows * cols, 1, CvType.CV_32F, new Scalar(0));
        for (int i = 0; i < cols; i ++){
            for (int j = 0; j < rows; j++){
                result[i*rows + j] = arr[j][i];
            }
        }
        return result;
    }
}
