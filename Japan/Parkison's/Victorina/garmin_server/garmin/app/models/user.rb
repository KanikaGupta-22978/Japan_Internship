class User < ApplicationRecord
  devise :database_authenticatable, :masqueradable,
         :rememberable, :omniauthable, omniauth_providers: %i[garmin]

  has_many :dailies
  has_many :sleeps
  has_many :stresses
  has_many :epochs
  has_many :pulse_oxes
  has_many :respirations
  has_many :heart_rate_variabilities

  def self.from_omniauth(auth)
    user = find_or_initialize_by(provider: auth.provider,
                                user_access_token: auth.credentials.token).tap do |user|
        user.user_access_token = auth.credentials.token
        user.user_access_token_secret = auth.credentials.secret
    end
    user.save
    user
  end
end
