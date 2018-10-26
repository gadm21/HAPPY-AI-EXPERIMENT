.PHONY: migrate
migrate:
	DATABASE_URL=sqlite:////tmp/ai_api.db FLASK_APP=run:app flask db migrate

.PHONY: init
init:
	DATABASE_URL=sqlite:////tmp/ai_api.db FLASK_APP=run:app flask db init
	$(MAKE) migrate

.PHONY: test
test:
	DATABASE_URL=sqlite:////tmp/ai_api_tests.db APP_MODE=testing pytest

.PHONY: run
run:
	DATABASE_URL=sqlite:////tmp/ai_api.db FLASK_APP=run:app flask run
