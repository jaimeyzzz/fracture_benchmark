python initialize2.py $1 $2 $3
python simulate2.py $1 $2 $3
ffmpeg -r 30 -start_number 0 -i output/images/$1_$2/%04d.jpg -vframes $3 -vcodec libx264 -crf 25 -pix_fmt yuv420p -y output/$1_$22.mp4