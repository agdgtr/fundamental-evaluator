from __future__ import annotations

from flask import Flask, render_template, request

from .service import FundamentalScorerService


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    service = FundamentalScorerService()

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None
        form_state = {
            "symbol": "MSFT",
            "current_hold": 0,
            "macro_score": 50,
            "sec_name": "",
            "sec_email": "",
        }

        if request.method == "POST":
            form_state["symbol"] = request.form.get("symbol", "MSFT").strip().upper()
            form_state["sec_name"] = request.form.get("sec_name", "").strip()
            form_state["sec_email"] = request.form.get("sec_email", "").strip()
            try:
                form_state["current_hold"] = float(request.form.get("current_hold", "0") or 0)
                form_state["macro_score"] = float(request.form.get("macro_score", "50") or 50)
                if not form_state["sec_name"] or not form_state["sec_email"]:
                    raise ValueError("Enter the SEC request name and email once so this browser can identify itself to SEC EDGAR.")
                result = service.analyze(
                    form_state["symbol"],
                    current_hold=form_state["current_hold"],
                    macro_score=form_state["macro_score"],
                    sec_name=form_state["sec_name"],
                    sec_email=form_state["sec_email"],
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                error = str(exc)

        return render_template("index.html", result=result, error=error, form=form_state)

    return app
