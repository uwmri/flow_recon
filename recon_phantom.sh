#! /bin/sh

cd /data/users/groberts/SMS_Testing/Exam5450

cd SMS_MB2
# sms_2dpc_recon ScanArchive_608262WIMRMR1_20220808_174747994.h5
llr_recon_flow.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 1 --smap_type walsh --sms_factor 1
mv FullRecon.h5 InPhase.h5
llr_recon_flow.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 1 --smap_type lowres --sms_factor 2
mv FullRecon.h5 OutPhase.h5

# cd ../SMS_MB3

# cdSMS_MB4

# cd /export/home/groberts/CODE/PRECON/flow_recon/