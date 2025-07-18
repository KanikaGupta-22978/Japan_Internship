class CreateSurveyResponses < ActiveRecord::Migration[5.2]
  def change
    create_table :survey_responses do |t|
      t.references :user, foreign_key: true
      t.jsonb :answers, null: false, default: {}

      t.timestamps
    end
  end
end
