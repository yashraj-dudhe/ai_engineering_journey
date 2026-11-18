import logging
import sys

root = logging.getLogger()
print(f"root level: {root.level} = {logging.getLevelName(root.level)}") 

log = logging.getLogger("demo")
log.info("you will NOT see this")     # 20 < 30, discarded
log.warning("you WILL see this")     # 30 >= 30, emitted

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s | %(message)s",
    stream=sys.stdout,  # default is stderr; pin it so this interleaves with print()
    force=True,         # basicConfig is a NO-OP if a handler already exists
)

log.info("now this appears, and nothing at the call site changed")


for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    print(f"{name:<8} = {getattr(logging, name)}")


class Expensive:
    def __str__(self):
        print("    <-- __str__ ran, so the string WAS built")
        return "expensive"


log.setLevel(logging.INFO)          # DEBUG is now switched off

print("f-string form:")
log.debug(f"value is {Expensive()}")   # built anyway, then thrown away

print("argument form:")
log.debug("value is %s", Expensive())  # __str__ never runs

try:
    1 / 0
except ZeroDivisionError:
    log.exception("division failed")   # ERROR + the full traceback