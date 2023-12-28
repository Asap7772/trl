# https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket
# https://cloud.google.com/storage/docs/gcsfuse-mount#control_access_permissions_to_the_mount_point
bucket_name="rlhf_bucket_anikait"
mkdir -p "$HOME/mount-folder"
sudo mount -t gcsfuse -o allow_other,implicit_dirs "$bucket_name" "$HOME/mount-folder"
# mount -t gcsfuse -o allow_other,implicit_dirs,writeback_cache_max_size=1024,attr_timeout=120,entry_timeout=120,dir_cache_max_size=1024,noatime,explicit-dirs,uid=1000,gid=1000 "$bucket_name" "$HOME/mount-folder"