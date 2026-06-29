# Postgres with a self-signed cert so connections are TLS-encrypted in transit
# (the app connects with sslmode=require). Certs live outside the data volume so
# they persist with the image, not the volume.
FROM postgres:16-bookworm

RUN mkdir -p /etc/postgresql \
 && openssl req -new -x509 -days 3650 -nodes -text \
        -subj "/CN=llmgames-db" \
        -out /etc/postgresql/server.crt \
        -keyout /etc/postgresql/server.key \
 && chmod 600 /etc/postgresql/server.key \
 && chown postgres:postgres /etc/postgresql/server.key /etc/postgresql/server.crt
