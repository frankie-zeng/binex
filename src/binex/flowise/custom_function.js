"use strict";

const fs = require("fs");
const vm = require("vm");

const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const sandbox = {
  $input: payload.input,
  $flow: payload.flow,
  $vars: payload.vars || {},
  console,
};
for (const [key, value] of Object.entries(payload.extra || {})) {
  sandbox[key] = value;
}
vm.createContext(sandbox);

function finish(body, code) {
  process.stdout.write(JSON.stringify(body));
  if (code) process.exit(code);
}

try {
  const result = vm.runInContext(payload.code, sandbox, {
    timeout: payload.timeoutMs || 5000,
  });
  Promise.resolve(result)
    .then((value) => {
      finish({ ok: true, result: value === undefined ? null : value });
    })
    .catch((error) => {
      finish({ ok: false, error: String((error && error.stack) || error) }, 1);
    });
} catch (error) {
  finish({ ok: false, error: String((error && error.stack) || error) }, 1);
}
