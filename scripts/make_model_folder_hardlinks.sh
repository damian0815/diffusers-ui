transformer=$1
output=$2
base=$3
if [ "x$base" == "x" ]; then
	base=/workspace/models/Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers
fi

if [[ "x$transformer" == "x" || "x$output" == "x" ]]; then
	echo "usage: $0 <transformer file> <output> [<base model>]"
	exit 1 
fi

if [ ! -e $base ]; then
	echo "$base does not exist"
	exit 1
fi

if [ ! -e "$transformer" ]; then
	echo "$transformer does not exist"
	exit 1
fi

if [ -e "$output" ]; then
	echo "$output already exists"
	exit 1
fi

set -x
out_transformer="$output"/transformer/diffusion_pytorch_model.safetensors
cp -la "$base" "$output"
rm "$out_transformer"
ln "$transformer" "$out_transformer"

