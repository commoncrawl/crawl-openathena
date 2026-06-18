# bot-blocking in the Open Athena pilot crawl

## Intro

I revived some code from 2018 that fingerprints webservers.

Jargon: A "quad" is what we call the 4 related hosts of:
- http://example.com/
- https://example.com/
- http://www.example.com/
- https://www.example.com/

Since robots.txt is the first thing fetched in the crawl, there are
often redirects between these 4 things.

We're interested in if we got a robots.txt at all, did it disallow us,
and was Cloudflare seen at the scene of the crime.

This crawl had 9757 quads, of which 256 (2.6%) had Cloudflare fingerprints.

## Overall

- 6638 quads had a robots.txt (status 200) (68%)
- 2118 quads had no robots.txt (404 or 410) (22%)
- 523 quads redirected us outside the quad, we did not analyze those (5%)

## Bot defenses

- 729 quads sent us only error codes instead of 200/404/410 (7.5%) -- bot blocking?
  - of these, 179 had Cloudflare fingerprints
    - 162 of these were bot blocking
    - 17 were errors related to the "origin host" being down -- not bot blocking

## robots.txt

Of 6638 quads that sent us a robots.txt:

- 120 disallowed all bots -- mostly due to an "allowlist" strategy
  -  0 of those had Cloudflare fingerprints
- 425 disallowed GPTBot
  - 192 of these had Cloudflare fingerprints
- 299 disallowed CCBot
  - 192 of these had Cloudflare fingerprints

## Summary

- 729 quads of bot blocking is 7.5% of 9757 quads
- 120+299 CCBot robots.txt disallowed 4.3% of 9757 quads
- 64% of CCBot robots.txt disallowed has Cloudflare fingerprints

## Future TODO:

What technologies other than Cloudflare are commonly blocking CCBot?
