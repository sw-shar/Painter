build:
	docker build . -t flaskimage
	
run:
	docker run --rm --name=flaskcontainer -it -p 5000:5000 flaskimage
	
bash:
	docker run --rm --name=flaskcontainer -it flaskimage bash
