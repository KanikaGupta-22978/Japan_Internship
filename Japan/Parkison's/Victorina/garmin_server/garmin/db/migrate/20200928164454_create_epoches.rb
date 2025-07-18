class CreateEpoches < ActiveRecord::Migration[5.2]
  def change
    create_table :epoches do |t|
      t.references :user, foreign_key: true
      t.date :calendar_date
      t.jsonb :data, null: false, default: {}

      t.timestamps
    end
  end
end
