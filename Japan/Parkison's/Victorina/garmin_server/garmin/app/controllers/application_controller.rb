class ApplicationController < ActionController::Base
	before_action :masquerade_user!

	def default_url_options(options={})
		{ locale: I18n.locale }
	end
end
