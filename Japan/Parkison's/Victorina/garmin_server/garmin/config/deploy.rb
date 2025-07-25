# config valid for current version and patch releases of Capistrano
lock "~> 3.11.0"

set :rvm_ruby_version, '2.5.8'

set :application, "garmin"
set :repo_url, "git@github.com:ShibataLab/garmin.git"

# Default branch is :master
# set :branch, `git rev-parse --abbrev-ref HEAD`
if ENV['branch']
  set :branch, ENV['branch']
else
	ask :branch, `git rev-parse --abbrev-ref HEAD`.chomp
end

# Default deploy_to directory is /var/www/my_app_name
set :deploy_to, "/home/ubuntu/garmin"

# Default value for :format is :airbrussh.
# set :format, :airbrussh

# You can configure the Airbrussh format using :format_options.
# These are the defaults.
# set :format_options, command_output: true, log_file: "log/capistrano.log", color: :auto, truncate: :auto

# Default value for :pty is false
# set :pty, true

# Default value for :linked_files is []
append :linked_files, "config/database.yml", ".env.production", "config/master.key", 
  "config/sidekiq.yml", "docker-compose.yml"

# Default value for linked_dirs is []
append :linked_dirs, "log", 
  "tmp/pids", "tmp/cache", "tmp/sockets", 
  "vendor/bundle", "public/system", "storage"
  # "models", "fastapi"

# Default value for default_env is {}
# set :default_env, { path: "/opt/ruby/bin:$PATH" }

# Default value for local_user is ENV['USER']
set :local_user, -> { `git config user.name`.chomp }

# Default value for keep_releases is 5
# set :keep_releases, 5

# Uncomment the following to require manually verifying the host key before first deploy.
# set :ssh_options, verify_host_key: :secure

set :yarn_flags, '--production --no-progress'

# namespace :deploy do
#   task :sidekiq do
#     on roles(:app) do
#       execute :sudo, :systemctl, :restart, :sidekiq
#     end
#   end
# end

# after "deploy", "deploy:sidekiq"

set :ssh_options, forward_agent: true,
	keys: Dir['config/keys/*']