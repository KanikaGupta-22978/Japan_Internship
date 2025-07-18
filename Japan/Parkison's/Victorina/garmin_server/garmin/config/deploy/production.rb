server 'ec2-52-44-200-52.compute-1.amazonaws.com', user: 'ubuntu', roles: %w{app db web}

set :rails_env, :production

set :deploy_to, '/home/ubuntu/garmin'