"""results.json -> report.html (self-contained; data embedded, no network)."""
import json
import html

R = json.load(open("results.json", encoding="utf-8"))
TEMPLATE = open("report_template.html", encoding="utf-8").read()
runs = [{"name": n, **{k: h[k] for k in ("step", "train_loss", "val_loss", "val_tok_acc", "val_seq_acc", "flush")},
         "sec_per_step": h["sec_per_step"], "drift": h["weight_drift"], "config": h["config"]}
        for n, h in R["runs"].items()]
data = json.dumps({"steps": R["steps"], "gpu": R["gpu"], "runs": runs}, ensure_ascii=False)
# </script> cannot appear inside JSON from our own names, but escape defensively.
data = data.replace("</", "<\\/")
out = TEMPLATE.replace("__DATA__", data).replace("__GPU__", html.escape(R["gpu"]))
open("report.html", "w", encoding="utf-8").write(out)
print("wrote report.html")
