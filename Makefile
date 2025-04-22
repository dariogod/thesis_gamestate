.PHONY: dockerbuild
dockerbuild:
	docker compose build

.PHONY: dockerrun
dockerrun:
	bash -c "sudo docker compose up"

.PHONY: docker
docker:
	$(MAKE) dockerbuild
	$(MAKE) dockerrun

.PHONY: udocker
udocker:
	udocker pull $(shell docker compose config | grep "image:" | awk '{print $$2}')
	udocker create --name=tracklab $(shell docker compose config | grep "image:" | awk '{print $$2}')
	udocker run --volume="$(PWD)/thesis_sn-gamestate:/app/thesis_sn-gamestate" \
		--volume="$(PWD)/thesis_tracklab:/app/thesis_tracklab" \
		--volume="$(PWD)/data:/app/data" \
		--volume="$(PWD)/outputs:/app/outputs" \
		--env="NVIDIA_VISIBLE_DEVICES=all" \
		--publish="8000:8000" \
		tracklab

