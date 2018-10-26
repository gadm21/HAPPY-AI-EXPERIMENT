.PHONY: test
test:
	$(MAKE) -C server test
	DATABASE_URL=sqlite:////tmp/ai_api_tests.db APP_MODE=testing pytest ./server/tests ./server/tests

.PHONY: run
.DEFAULT: run
run:
	$(MAKE) -C server run

.PHONY: migrate
migrate:
	$(MAKE) -C server migrate
