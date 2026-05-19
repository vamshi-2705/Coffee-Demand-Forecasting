mkdir -p .streamlit

cat << EOF > .streamlit/config.toml
[server]
port = \$PORT
enableCORS = false
headless = true
EOF
