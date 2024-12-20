#!/bin/bash
cd dataprepare/training
unzip -q -d . training_dataset.zip
mv training_dataset/image .
mv training_dataset/mask .
# rm training_dataset.zip
rm -rf training_dataset
rm -rf __MACOSX
ls

cd ../testing

unzip -q -d . testing_dataset.zip
mv testing_dataset/image .
mv testing_dataset/mask .
# rm testing_dataset.zip
rm -rf testing_dataset
rm -rf __MACOSX
ls

cd ../../