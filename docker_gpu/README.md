Build the docker image
```bash
cd docker_gpu
sudo docker build -t fasterrcnn_gpu .
```

Run the docker container
when pwd is `waste_reduction_data_science`
```bash
sudo docker run -d -it \
--name fasterrcnn_gpu \
--volume $(pwd):/code \
--net=host \
--restart unless-stopped \
--runtime=nvidia \
--entrypoint "/bin/bash" fasterrcnn_gpu:latest \
-c "cd app && uwsgi --ini uwsgi.ini --processes 1"
```

Container logs
```bash
sudo docker logs --tail 100 fasterrcnn_gpu 
```

Check API endpoint
```bash
curl -X GET -H 'Accept: */*' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'User-Agent: python-requests/2.25.1' http://localhost:5000/detect_trucks
```