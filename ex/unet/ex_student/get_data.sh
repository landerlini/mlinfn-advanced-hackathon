save_dir=`pwd`
save_dir=${save_dir%cxr-hackathon*}
save_dir=$save_dir"Shezen_128128"

if [ -d "$save_dir" ]; then
    echo "data already downloaded"
else
    wget https://pandora.infn.it/public/f10e06/dl/Shezen_128128.zip
    unzip Shezen_128128.zip
    echo $save_dir
    rm Shezen_128128.zip
    mv Shezen_128128 $save_dir
fi