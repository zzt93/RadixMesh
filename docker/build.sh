#
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # 定位到 docker目录
cd "$DIR/.."

docker build -f docker/Dockerfile -t radixmesh .
docker run --rm -it -v "$(pwd)/python":/app radixmesh