docker build -t tahaouarrak/rvc:latest -t tahaouarrak/rvc:$GITHUB_SHA .
docker push tahaouarrak/rvc:latest
docker push tahaouarrak/rvc:$GITHUB_SHA
