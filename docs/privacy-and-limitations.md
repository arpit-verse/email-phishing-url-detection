# Privacy, safety, and limitations

## Data flow

Email text and URL strings entered in the analysis forms are sent to the configured API. The API does not return raw email content, normalized content, recognized email tokens, or extracted URL features. The URL endpoint does not request the URL from the network.

The application code does not add database persistence or authentication. Successful scan history is held in React memory for the current page session; overview counts, activity, and sample history remain demo data. The application cannot guarantee how a hosting provider, browser, proxy, or network records traffic.

Do not submit confidential, personal, or regulated email content to a public deployment unless the service owner has approved data handling and access controls. The current API has no login or per-user access control.

## Model limitations

- Email training and dataset source provenance are incomplete; the holdout was used in model selection.
- URL features are lexical and structural only. They do not inspect page content, TLS state, registration age, reputation feeds, or redirects.
- URL test rows share many registrable domains with training rows.
- Probability calibration and performance under distribution shift have not been established.
- Global feature weights do not explain an individual result. The API returns no local attribution.
- These models can produce false positives and false negatives and are not security guarantees.

## Release limitations

Raw datasets, serialized model binaries, row-level errors, and original Git history are excluded. Dataset redistribution terms and a project source-code license have not been established in the available evidence. Public availability of model artifacts does not establish training-data redistribution rights.