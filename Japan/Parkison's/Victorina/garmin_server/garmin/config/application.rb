require_relative 'boot'

require 'rails/all'

# Require the gems listed in Gemfile, including any gems
# you've limited to :test, :development, or :production.
Bundler.require(*Rails.groups)

module App
  class Application < Rails::Application
    # Initialize configuration defaults for originally generated Rails version.
    config.load_defaults 5.2

    # Settings in config/environments/* take precedence over those specified here.
    # Application configuration can go into files in config/initializers
    # -- all .rb files in that directory are automatically loaded after loading
    # the framework and any gems in your application.
    config.active_record.belongs_to_required_by_default = false

    config.active_job.queue_adapter = :sidekiq

    config.generators.javascript_engine = :js
    config.generators.test_framework = nil
    config.generators.jbuilder = false
    config.generators.serializer = false

    config.time_zone = 'Asia/Tokyo'
  end
end
