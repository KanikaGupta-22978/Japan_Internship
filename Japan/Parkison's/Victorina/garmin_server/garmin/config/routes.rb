Rails.application.routes.draw do
  
  devise_for :users, controllers: { 
    omniauth_callbacks: 'users/omniauth_callbacks',
    masquerades: 'admin/masquerades'
  }
  devise_for :admin_users, ActiveAdmin::Devise.config
  ActiveAdmin.routes(self)

  resources :users, only: [:show]
  scope "/:locale", locale: /en|ja/ do
  end

  namespace :api do
  	post 'dailies', to: 'dailies#index'
  	post 'sleeps', to: 'sleeps#index'
  	post 'stresses', to: 'stresses#index'
    post 'epochs', to: 'epochs#index'
    post 'pulse_oxes', to: 'pulse_oxes#index'
    post 'respirations', to: 'respirations#index'
    post 'heart_rate_variabilities', to: 'heart_rate_variabilities#index'
    get 'predict/:id/timestamp/:timestamp', to: 'predicts#index'
  end

  root to: 'users#show'
end
